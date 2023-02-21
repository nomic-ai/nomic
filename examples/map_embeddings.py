from nomic import atlas
import numpy as np

num_embeddings = 10000
embeddings = np.random.rand(num_embeddings, 512)

response = atlas.map_embeddings(embeddings=embeddings)
print(response)