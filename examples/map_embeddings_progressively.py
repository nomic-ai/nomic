from nomic import atlas
import numpy as np

from nomic import atlas
import numpy as np

num_embeddings = 1000
embeddings = np.random.rand(num_embeddings, 10)
data = [{'upload': '1'} for i in range(len(embeddings))]

project = atlas.map_embeddings(embeddings=embeddings,
                               data=data,
                               name='A Map That Gets Updated',
                               colorable_fields=['upload'])
map = project.get_map('A Map That Gets Updated')
print(map)

# embeddings with shifted mean.
embeddings += np.ones(shape=(num_embeddings, 10))
data = [{'upload': '2'} for i in range(len(embeddings))]

with project.wait_for_project_lock() as project:
    project.add_embeddings(embeddings=embeddings, data=data)
    project.rebuild_maps()