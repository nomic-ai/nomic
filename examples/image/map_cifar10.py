from nomic import embed
from nomic import atlas
from PIL import Image
import numpy as np

from datasets import load_dataset

dataset = load_dataset("cifar10")['train']
labels = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

images = []
datums = []

for idx, image in enumerate(dataset):
    images.append(image['img'])
    datums.append({'id': str(idx),
                   'label': labels[image['label']]})
    #only map 1000 images for demo purposes
    if idx >= 1000:
        break

output = embed.images(images=images)

embeddings = np.array(output['embeddings'])

atlas.map_data(embeddings=embeddings,
               identifier='CIFAR',
               data=datums,
               id_field='id',
               topic_model=False)

