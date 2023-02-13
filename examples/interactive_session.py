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

# from nomic import atlas
# import numpy as np
#
# num_embeddings = 10000
# embeddings = np.random.rand(num_embeddings, 256)
#
# project = atlas.map_embeddings(embeddings=embeddings)
# print(project.maps)
#
# exit()


# from nomic import atlas
# import numpy as np
#
# num_embeddings = 1000
# embeddings = np.random.rand(num_embeddings, 10)
# data = [{'upload': '1', 'id': i} for i in range(len(embeddings))]
#
# project = atlas.map_embeddings(embeddings=embeddings,
#                                data=data,
#                                id_field='id',
#                                map_name='A Map That Gets Updated',
#                                colorable_fields=['upload'],
#                                reset_project_if_exists=True)
# map = project.get_map('A Map That Gets Updated')
# print(map)
#
# total_datums = project.total_datums
# # embeddings with shifted mean.
# embeddings += np.ones(shape=(num_embeddings, 10))
# data = [{'upload': '2', 'id': total_datums+i} for i in range(len(embeddings))]
#
# with project.block_until_accepting_data():
#     project.add_embeddings(embeddings=embeddings, data=data)
#     project.rebuild_maps()
#
#
#
# with project.block_until_accepting_data():
#     project.delete_data(ids=[i for i in range(1100, 2000)])
#     project.rebuild_maps()
#
#
# print(project.get_data(ids=[i for i in range(0, 1000)]))


# from nomic import atlas
# import numpy as np
# from datasets import load_dataset
#
# np.random.seed(0)  # so your map has the same points sampled.
#
# dataset = load_dataset('ag_news')['train']
#
# max_documents = 10000
# subset_idxs = np.random.randint(len(dataset), size=max_documents).tolist()
# documents = [dataset[i] for i in subset_idxs]
# for i in range(len(documents)):
#     documents[i]['id'] = i
#
# project = atlas.map_text(data=documents,
#                          id_field='id',
#                          indexed_field='text',
#                          map_name='News 10k Labeling Example',
#                          map_description='10k News Articles for Labeling'
#                          )
# print(project.maps)

project = AtlasProject(name='News 10k Labeling Example')

tags = project.get_tags()
for tag, datum_ids in tags.items():
    print(tag, "Count:", len(datum_ids), datum_ids[:10])

print(project.get_data(ids=tags['sports'][:2]))

project.delete_data(ids=tags['sports'])
project.rebuild_maps()