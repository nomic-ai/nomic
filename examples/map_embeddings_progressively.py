from nomic import AtlasClient
import numpy as np

atlas = AtlasClient()

num_embeddings = 1000
embeddings = np.random.rand(num_embeddings, 256)
data = [{'id': i} for i in range(len(embeddings))]

first_upload_embeddings = embeddings[:500, :]
second_upload_embeddings = embeddings[500:, :]
first_upload_data = data[:500]
second_upload_data = data[500:]

response = atlas.map_embeddings(embeddings=first_upload_embeddings,
                                data=first_upload_data,
                                is_public=True)

print('First upload response: ', response)
project_id = response['project_id']

response = atlas.update_maps(project_id=project_id,
                             embeddings=second_upload_embeddings,
                             data=second_upload_data)
print('Second upload response: ', response)
