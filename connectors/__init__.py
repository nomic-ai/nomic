from nomic.connectors import huggingface_connecter

atlas_dataset = huggingface_connecter.load('aaa/bbb')

atlas_dataset.create_index(...)