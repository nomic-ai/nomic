from nomic import AtlasClient
import numpy as np

atlas = AtlasClient()

num_embeddings = 10000
embeddings = np.random.rand(num_embeddings, 256)
data = [{'id': i} for i in range(len(embeddings))]

project_name = 'test'
index_name = 'test'

atlas.create_project(project_name=project_name,
                     description=project_name,
                     unique_id_field='id',
                     modality='embedding')

print("Adding Data")
atlas.add_embeddings(project_name=project_name,
                     embeddings=embeddings,
                     data=data)

print("Organizing Data")
atlas.create_index(project_name=project_name, index_name=index_name)
