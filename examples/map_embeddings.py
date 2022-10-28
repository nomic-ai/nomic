from nomic import AtlasClient
import numpy as np

atlas = AtlasClient()

num_embeddings = 10000
embeddings = np.random.rand(num_embeddings, 256)

response = atlas.map_embeddings(embeddings=embeddings)
print(response)