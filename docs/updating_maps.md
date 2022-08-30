# Updating Maps
The nomic client also allows you to update existing maps with new data.
When you upload new data to a project, every index in that project will be updated with that new data.
A minimal example of this for text projects is shown below:

=== "Progressive Upload Example"
```
from nomic import AtlasClient
import numpy as np
from datasets import load_dataset

atlas = AtlasClient()

dataset = load_dataset('wikipedia', '20220301.en')['train']

max_documents = 1000
subset_idxs = np.random.randint(len(dataset), size=max_documents).tolist()
documents = [dataset[i] for i in subset_idxs]

for idx, document in enumerate(documents):
    document['id'] = idx

first_upload = documents[:500]
second_upload = documents[500:]

response = atlas.map_text(data=first_upload,
                          id_field='id',
                          indexed_field='text',
                          is_public=True,
                          num_workers=10)

print('First upload response: ', response)
project_id = response['project_id']

response = atlas.update_maps(project_id=project_id,
                             data=second_upload)

print('Second upload response: ', response)

```

=== "Output"
```
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.04it/s]
2022-08-26 10:54:50.116 | INFO     | nomic.atlas:create_project:124 - Creating project `marked-quilt` in organization `brandon`
2022-08-26 10:54:52.271 | INFO     | nomic.atlas:map_text:701 - Uploading text to Nomic's neural database Atlas.
1it [00:04,  4.04s/it]
2022-08-26 10:54:56.316 | INFO     | nomic.atlas:map_text:713 - Text upload succeeded.
2022-08-26 10:54:57.948 | INFO     | nomic.atlas:create_index:400 - Created map `dizzy-hamburger`: https://staging-atlas.nomic.ai/map/778ba906-0348-4d1a-a59e-e5ff752c6bc8/c7a2c226-65f9-447c-9bc6-9c66bd59299e
First upload response:  {'map': 'https://staging-atlas.nomic.ai/map/778ba906-0348-4d1a-a59e-e5ff752c6bc8/c7a2c226-65f9-447c-9bc6-9c66bd59299e', 'job_id': '4f33ec17-dba2-4ca9-9f81-a7bc303ee570', 'index_id': '9ae5f720-2fa5-4905-9a2a-b8874e1cf081', 'project_id': '778ba906-0348-4d1a-a59e-e5ff752c6bc8'}
2022-08-26 10:54:58.348 | INFO     | nomic.atlas:update_maps:606 - Uploading data to Nomic's neural database Atlas.
1it [00:05,  5.60s/it]
2022-08-26 10:55:03.953 | INFO     | nomic.atlas:update_maps:626 - Upload succeeded.
Second upload response:  ['daf2c1df-6041-482e-9927-1a930c430492']
```
