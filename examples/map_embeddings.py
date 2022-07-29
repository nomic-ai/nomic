from nomic import AtlasClient
import numpy as np

atlas = AtlasClient()

num_embeddings = 10000
embeddings = np.random.rand(num_embeddings, 256)
data = [{'id': i} for i in range(len(embeddings))]

response = atlas.map_embeddings(embeddings=embeddings,
                                data=data,
                                id_field='id',
                                is_public=True)
print(response)

"""
map='https://atlas.nomic.ai/map/ff44fa13-a21b-4b7f-b850-a66bbc0b03de/d46c16cb-d8af-4626-8d2e-954fc1c4e04f'
job_id='4bb491fa-2dba-4461-8fbc-6df0f14ae5ae'
index_id='38fb9123-14c8-4ecb-acca-e22a81a1221b'
"""