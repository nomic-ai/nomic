from nomic.connectors import huggingface_connecter
import logging

# Example source url: https://huggingface.co/datasets/allenai/quartz
#Takes last two parts of url to get allenai/quartz
atlas_dataset = huggingface_connecter.load('allenai/quartz')

atlas_dataset.create_index(topic_model=True, embedding_model='NomicEmbed') 

logging.info("Atlas dataset has been loaded and indexed successfully.")
