from nomic import atlas
import numpy as np
from datasets import load_dataset

dataset = load_dataset('ag_news')['train']

max_documents = 1000
subset_idxs = np.random.randint(len(dataset), size=max_documents).tolist()
documents = [dataset[i] for i in subset_idxs]

first_upload = documents[:500]
second_upload = documents[500:]

project = atlas.map_text(data=first_upload,
                          indexed_field='text',
                          name='News 1k Example Progressive',
                        )

print(project.maps)

with project.wait_for_project_lock():
    project.add_text(data=second_upload)
    project.rebuild_maps()
