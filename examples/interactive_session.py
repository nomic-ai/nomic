from nomic import AtlasClient
import numpy as np

atlas = AtlasClient()

num_embeddings = 10000
embeddings = np.random.rand(num_embeddings, 256)
data = [{'id': i} for i in range(len(embeddings))]

project_name = 'My Project'
index_name = 'My Index'

project_id = atlas.create_project(project_name=project_name,
                     description=project_name,
                     modality='embedding')

print(f"Adding embeddings to project: {project_id}")
atlas.add_embeddings(project_id=project_id,
                     embeddings=embeddings,
                     data=data)

print(f"Organizing embeddings in project: {project_id}")
response = atlas.create_index(project_id=project_id, index_name=index_name)
print(response)

print(atlas._get_index_job(job_id=response.job_id))
