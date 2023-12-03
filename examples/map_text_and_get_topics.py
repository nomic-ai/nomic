from nomic import atlas
import numpy as np
from nomic import AtlasDataset
from pprint import pprint
from datasets import load_dataset

dataset = load_dataset('ag_news')['train']

max_documents = 10000
subset_idxs = np.random.choice(len(dataset), size=max_documents, replace=False).tolist()
documents = [dataset[i] for i in subset_idxs]

project = atlas.map_text(data=documents,
                          indexed_field='text',
                          name='News 10k For Topic Extraction',
                          colorable_fields=['label'],
                          description='News 10k For Topic Extraction'
                          )



project.wait_for_project_lock()

# project = AtlasDataset(name='News 10k For Topic Extraction')

from pprint import pprint
pprint(project.maps[0].topics.group_by_topic(topic_depth=1)[0])


