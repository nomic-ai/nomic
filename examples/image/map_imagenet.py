from nomic import embed
from nomic import atlas
import base64
import os
import numpy as np
import time

from datasets import load_dataset

# dataset = load_dataset("imagenet-1k")['train']


rootdir = '/home/ubuntu/remote_dev/imagenet'

images = []
datums = []
idx = 0
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        label = subdir.split('/')[-1]
        datums.append({'id': str(len(images)),
                       'label': label,
                       'url': f'https://nomic-research.s3.us-east-2.amazonaws.com/imagenet/data/{label}/{file}'})
        images.append(os.path.join(subdir, file))
    if len(images) >= 1000:
        break


# for idx, image in enumerate(dataset):
#     images.append(image['img'])
#     datums.append({'id': str(idx), 'label': image['label']})
#     if idx >= 10000:
#         break

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
                     build_topic_model=True
                     )

