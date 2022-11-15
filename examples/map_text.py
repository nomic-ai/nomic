from nomic import AtlasClient
import numpy as np
from datasets import load_dataset

atlas = AtlasClient()

dataset = load_dataset('wikipedia', '20220301.en')['train']

max_documents = 10000
subset_idxs = np.random.randint(len(dataset), size=max_documents).tolist()
documents = [dataset[i] for i in subset_idxs]

response = atlas.map_text(data=documents,
                          indexed_field='text',
                          is_public=True,
                          map_name='Wiki 10K',
                          map_description='A 10,000 point sample of the huggingface wikipedia dataset embedded with Nomic\'s Embed v0.0.13 model.',
                          organization_name=None, #defaults to your current user.
                          num_workers=10)
print(response)