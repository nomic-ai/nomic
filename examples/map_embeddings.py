from nomic import atlas
from nomic import AtlasDataset
import numpy as np

# num_embeddings = 10000
# embeddings = np.random.rand(num_embeddings, 512)
#
# response = atlas.map_embeddings(embeddings=embeddings)
# print(response)

x = AtlasDataset('andriy/ai-summit-map-1')
print(x)

# print(x.maps[0].embeddings.latent)