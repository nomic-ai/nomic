from nomic import atlas
import numpy as np
from datasets import load_dataset
dataset = load_dataset('ag_news')['train']

np.random.seed(0)
max_documents = 25000
subset_idxs = np.random.randint(len(dataset), size=max_documents).tolist()
documents = [dataset[i] for i in subset_idxs]

project = atlas.map_text(data=documents,
                          indexed_field='text',
                          name='News Dataset 25k',
                          colorable_fields=['label'],
                          description='News Dataset 25k'
                          )


with project.wait_for_project_lock():
    map = project.get_map(name='News Dataset 25k')
    print(map.map_link)
    print(project.total_datums)

