from __future__ import annotations
import random
from numpy import stack
from pandas import DataFrame, read_csv
from cv2 import imread, resize, INTER_AREA
from torch import Tensor
from torchvision import transforms
import imgaug.augmenters as iaa
from transformers import DistilBertTokenizer
from modelbase.core.dataset import AbstractDFDataset, DataLoader


class DatasetBuilder:
    train_dataset: Flickr8kDataset
    validation_dataset: Flickr8kDataset
    train_dataloader: DataLoader
    validation_dataloader: DataLoader

    def __init__(self, batch_size: int = 2, num_workers: int = 0):
        self.batch_size = batch_size
        self.num_workers = num_workers

    def build_train_validation_dataset(self, csv_file: str, img_folder=''):
        data = read_csv(csv_file)  # .iloc[:200]  # DEBUG
        unique_image_set = set(data['image'].tolist())
        validation_img_set = set(random.sample(unique_image_set, int(len(unique_image_set) * 0.2)))
        train_img_set = unique_image_set - validation_img_set
        train_df = data[data['image'].isin(train_img_set)]
        validation_df = data[data['image'].isin(validation_img_set)]

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.train_dataset = Flickr8kDataset(train_df, tokenizer, img_folder)
        self.validation_dataset = Flickr8kDataset(validation_df, tokenizer, img_folder, doAugment=False)

        self.train_dataloader = self.train_dataset.get_dataloader(
            self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=False)
        self.validation_dataloader = self.validation_dataset.get_dataloader(
            self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False)


class Flickr8kDataset(AbstractDFDataset):
    """ Based on https://www.kaggle.com/datasets/aladdinpersson/flickr8kimagescaptions dataset
    """

    def __init__(self, data: DataFrame, tokenizer: DistilBertTokenizer,
                 img_folder='', tokenizer_max_length=200, img_size=(224, 224),
                 returnWithLabel=True, doAugment=False):
        super(Flickr8kDataset, self).__init__(returnWithLabel, doAugment)
        self.img_folder = img_folder
        self.data_list = data
        self.img_size = img_size
        captions = data['caption'].tolist()
        self.encoded_captions = tokenizer(captions, padding=True, truncation=True, max_length=tokenizer_max_length)

    def get_input_and_label(self, idx: int) -> dict:
        # process caption
        text_item = {
            'token_ids': self.encoded_captions['input_ids'][idx],
            'attention_mask': self.encoded_captions['attention_mask'][idx],
            'caption': self.data_list.iloc[idx]['caption']}

        # process image
        item = self.get_input_only(idx)
        item.update(text_item)
        return item

    def get_input_only(self, idx: int) -> dict:
        filename = f"{self.img_folder}/{self.data_list.iloc[idx]['image']}"
        img = imread(filename)[..., [2, 1, 0]]
        img = resize(img, self.img_size, interpolation=INTER_AREA)
        if self.doAugment:
            img = img_augmenter()([img])[0]
        return {'image': img}

    @classmethod
    def convert_img_to_tensor(
            cls, img: Tensor, img_max=255, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> Tensor:
        _trans_func = transforms.Normalize(mean, std)
        return _trans_func(img.permute(0, 3, 1, 2) / img_max)

    def collate_fn(self, batch: list):
        token_ids = Tensor([dic['token_ids'] for dic in batch]).long()
        attention_mask = Tensor([dic['attention_mask'] for dic in batch]).long()
        image = Tensor(stack([dic['image'] for dic in batch], axis=0))
        image = self.convert_img_to_tensor(image)
        return {'image': image, 'attention_mask': attention_mask, 'token_ids': token_ids}


def img_augmenter():
    # flake8: noqa: E731
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
        iaa.Crop(px=(1, 50), keep_size=True),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        sometimes(iaa.GaussianBlur(sigma=(0, 1.5))),
        sometimes(iaa.AdditiveGaussianNoise(scale=(0, 12))),
        sometimes(iaa.LinearContrast((0.5, 2.0), per_channel=0.5)),
        sometimes(iaa.MultiplyBrightness((0.8, 1.2))),
        sometimes(iaa.MultiplyHue((0.85, 1.15))),
        sometimes(iaa.MultiplySaturation((0.8, 1.2))),
        sometimes(iaa.Grayscale(alpha=(0.0, 1.0)))
    ])
    return lambda img: seq(images=img)
