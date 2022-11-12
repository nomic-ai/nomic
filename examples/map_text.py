from nomic import AtlasClient
import numpy as np
from datasets import load_dataset

atlas = AtlasClient()

dataset = load_dataset('news_commentary', 'en-es')['train']

max_documents = 10000
subset_idxs = np.random.randint(len(dataset), size=max_documents).tolist()

documents = [dataset[i] for i in subset_idxs]

data = []
for id, document in enumerate(documents):
    for key in document['translation']:
        data.append({'text': document['translation'][key], 'language': key, 'source_document_id': id})

response = atlas.map_text(data=data,
                          indexed_field='text',
                          is_public=True,
                          colorable_fields=['language'],
                          map_name='news_commentary english embedder',
                          organization_name=None, #defaults to your current user.
                          reset_project_if_exists=True,
                          num_workers=10)
print(response)
