from typing import Tuple
import torch
import torchvision
from transformers import DistilBertModel, DistilBertConfig
from transformers.modeling_outputs import BaseModelOutput
from modelbase.core.network import AbstractNetwork


class ImgResnet50Encoder(AbstractNetwork):
    def __init__(self):
        super(ImgResnet50Encoder, self).__init__()
        self.backbone = torch.nn.Sequential(torchvision.models.resnet50(pretrained=True))

    def forward(self, inp):
        x = self.backbone(inp)
        return x


class TextEncoder(AbstractNetwork):
    def __init__(self, model_name="distilbert-base-uncased", pretrained=True):
        super(TextEncoder, self).__init__()
        self.model = DistilBertModel.from_pretrained(model_name) if pretrained\
            else DistilBertModel(config=DistilBertConfig())
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output: BaseModelOutput = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class ProjectionHead(AbstractNetwork):
    def __init__(self, embedding_dim: int, projection_dim=256, dropout=0.1):
        super(ProjectionHead, self).__init__()
        self.projection = torch.nn.Linear(embedding_dim, projection_dim)
        self.gelu = torch.nn.GELU()
        self.fc = torch.nn.Linear(projection_dim, projection_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class ClipNet(AbstractNetwork):
    def __init__(self, image_embedding=1000, text_embedding=768):
        super(ClipNet, self).__init__()
        self.image_encoder = ImgResnet50Encoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)

    def forward(self, batch) -> Tuple[torch.Tensor]:
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(input_ids=batch["token_ids"], attention_mask=batch["attention_mask"])
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        return image_embeddings, text_embeddings
