from nomic import atlas, AtlasProject
import nomic
import numpy as np
from datasets import load_dataset
# dataset = load_dataset('ag_news')['train']
#
# np.random.seed(0)
# max_documents = 25000
# subset_idxs = np.random.randint(len(dataset), size=max_documents).tolist()
# documents = [dataset[i] for i in subset_idxs]
#
# project = atlas.map_text(data=documents,
#                           indexed_field='text',
#                           map_name='News Dataset 25k',
#                           colorable_fields=['label'],
#                           map_description='News Dataset 25k'
#                           )

# project = AtlasProject(name='News Dataset 25k')
# map = project.get_map(name='News Dataset 25k')
# print(map.map_link)
# print(project.total_datums)
#
# tags = project.get_tags()
# print(tags.keys())

from nomic import atlas
import numpy as np

num_embeddings = 1000
embeddings = np.random.rand(num_embeddings, 10)
data = [{'upload': '1'} for i in range(len(embeddings))]

project = atlas.map_embeddings(embeddings=embeddings,
                               data=data,
                               map_name='A Map That Gets Updated',
                               colorable_fields=['upload'])
map = project.get_map('A Map That Gets Updated')
print(map)

# embeddings with shifted mean.
embeddings += np.ones(shape=(num_embeddings, 10))
data = [{'upload': '2'} for i in range(len(embeddings))]

with project.block_until_accepting_data() as project:
    project.add_embeddings(embeddings=embeddings, data=data)
    project.rebuild_maps()
