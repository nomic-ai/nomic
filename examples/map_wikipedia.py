from nomic import atlas
import numpy as np
from datasets import load_dataset

dataset = load_dataset('wikipedia', '20220301.en')['train']

max_documents = 10000
subset_idxs = np.random.randint(len(dataset), size=max_documents).tolist()
documents = [dataset[i] for i in subset_idxs]

response = atlas.map_text(data=documents,
                          indexed_field='text',
                          name='Wiki 10K',
                          description='A 10,000 point sample of the huggingface wikipedia dataset embedded with Nomic\'s Embed v0.0.13 model.',
                          build_topic_model=True)
print(response)