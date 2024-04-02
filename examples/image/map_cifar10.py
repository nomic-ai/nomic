from nomic import embed
from nomic import atlas
from tqdm import tqdm
import numpy as np
import base64
from io import BytesIO

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

max_embeddings = 100_000

for idx, image in enumerate(tqdm(dataset)):
    images.append(image['img'])
    datums.append({'id': str(idx),
                   'label': labels[image['label']],
                  }
                  )

    if idx >= max_embeddings:
        break

output = embed.images(images=images)

embeddings = np.array(output['embeddings'])

atlas.map_data(embeddings=embeddings,
               identifier=f'zach/cifar-nomic-embed-vision-v1-{len(embeddings)}-with-images',
               data=datums,
               id_field='id',
               topic_model=False)

