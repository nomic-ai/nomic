from nomic import AtlasClient
import numpy as np
from datasets import load_dataset

atlas = AtlasClient()

dataset = load_dataset('wikipedia', '20220301.en')['train']

max_documents = 1000
subset_idxs = np.random.randint(len(dataset), size=max_documents).tolist()
documents = [dataset[i] for i in subset_idxs]

first_upload = documents[:500]
second_upload = documents[500:]

response = atlas.map_text(data=first_upload,
                          indexed_field='text',
                          is_public=True,
                          num_workers=10)

print('First upload response: ', response)
project_id = response['project_id']
response = atlas.update_maps(project_id=project_id,
                             data=second_upload)

print('Second upload response: ', response)
