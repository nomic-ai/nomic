from nomic import atlas
import numpy as np
from datasets import load_dataset

dataset = load_dataset('ag_news')['train']

max_documents = 1000
subset_idxs = np.random.choice(len(dataset), size=max_documents, replace=False).tolist()
documents = [dataset[i] for i in subset_idxs]

first_upload = documents[:500]
second_upload = documents[500:]

dataset = atlas.map_data(data=first_upload,
                          indexed_field='text',
                          identifier='News 1k Example Progressive',
                        )

print(dataset.maps)

with dataset.wait_for_dataset_lock():
    dataset.add_data(data=second_upload)
    dataset.rebuild_maps()
