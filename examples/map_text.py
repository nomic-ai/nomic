from nomic import atlas
import numpy as np
from datasets import load_dataset

dataset = load_dataset('ag_news')['train']

max_documents = 10000
subset_idxs = np.random.randint(len(dataset), size=max_documents).tolist()
documents = [dataset[i] for i in subset_idxs]

project = atlas.map_text(data=documents,
                          indexed_field='text',
                          map_name='News 10k Example',
                          colorable_fields=['label'],
                          map_description='News 10k Example from the ag_news dataset hosted on huggingface.'
                          )
print(project.maps)


