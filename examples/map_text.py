from nomic import atlas, AtlasDataset
import numpy as np
from datasets import load_dataset

dataset = load_dataset('ag_news')['train']

max_documents = 100000
subset_idxs = np.random.choice(len(dataset), size=max_documents, replace=False).tolist()
documents = [dataset[i] for i in subset_idxs]
documents = [{'id': i, 'text': doc['text']} for i, doc in enumerate(documents)]

dataset = AtlasDataset(identifier="nomic/test-ag-news", unique_id_field='id')

for start in range(0, len(documents), 10000):
    dataset.add_data(documents[start:start+10000])

dataset.create_index(indexed_field='text')


# project = atlas.map_data(data=documents,
#                           indexed_field='text',
#                           identifier='News 100k Example',
#                           description='News 100k Example from the ag_news dataset hosted on huggingface.'
#                           )
# print(project.maps)


