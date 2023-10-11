from nomic import embed
from nomic import atlas
import base64
from io import BytesIO
import numpy as np
import time

from datasets import load_dataset

dataset = load_dataset("mnist")['train']

images = []
datums = []
for idx, image in enumerate(dataset):
    images.append(image['image'])
    datums.append({'id': str(idx), 'label': str(image['label'])})
    if idx >= 10000:
        break

start = time.time()
output = embed.images(images=images)

print(time.time() - start)
print(output['usage'])

embeddings = np.array(output['embeddings'])

print(embeddings.shape)

atlas.map_embeddings(embeddings=embeddings,
                     data=datums,
                     id_field='id',
                     colorable_fields=['label'],
                     build_topic_model=False
                     )



