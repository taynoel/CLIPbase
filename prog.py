from modelbase.process_store.clip_proc import ProcessBuilder


if __name__ == '__main__':
    builder = ProcessBuilder()
    clip_process = builder.build_from_yaml('resources/clip_process.yaml')
    clip_process.train()
