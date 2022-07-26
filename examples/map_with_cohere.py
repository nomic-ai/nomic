from nomic import AtlasClient
from nomic import CohereEmbedder
import numpy as np

atlas = AtlasClient()

cohere_api_key = ''
embedder = CohereEmbedder(cohere_api_key=cohere_api_key)

texts = ['Hello how are you', 'I am here.']
embeddings = embedder.embed(texts=texts)

data = [{'text': text} for i, text in enumerate(texts)]

response = atlas.map_embeddings(embeddings=np.array(embeddings), data=data, is_public=True)
print(response)


