from nomic import atlas
import numpy as np

num_embeddings = 1000
embeddings = np.random.rand(num_embeddings, 10)
data = [{'upload': '1'} for i in range(len(embeddings))]

dataset = atlas.map_data(embeddings=embeddings,
                               data=data,
                               name='A Map That Gets Updated',
                               colorable_fields=['upload'])
map = dataset.get_map('A Map That Gets Updated')
print(map)

# embeddings with shifted mean.
embeddings += np.ones(shape=(num_embeddings, 10))
data = [{'upload': '2'} for i in range(len(embeddings))]

with dataset.wait_for_dataset_lock() as project:
    dataset.add_embeddings(embeddings=embeddings, data=data)
    dataset.rebuild_maps()