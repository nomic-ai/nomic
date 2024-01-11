from nomic import atlas
import numpy as np
from datasets import load_dataset
dataset = load_dataset('ag_news')['train']

np.random.seed(0)
max_documents = 25000
subset_idxs = np.random.choice(len(dataset), size=max_documents, replace=False).tolist()
documents = [dataset[i] for i in subset_idxs]

dataset = atlas.map_data(data=documents,
                          indexed_field='text',
                          identifier='News Dataset 25k',
                          colorable_fields=['label'],
                          description='News Dataset 25k'
                          )


with dataset.wait_for_dataset_lock():
    map = dataset.maps[0]
    print(map.map_link)
    print(dataset.total_datums)

