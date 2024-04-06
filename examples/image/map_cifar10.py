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

max_embeddings = 200_000

for idx, image in enumerate(tqdm(dataset)):
    images.append(image['img'])
    buffered = BytesIO()
    image['img'].save(buffered, format="PNG")

    b64img = base64.b64encode(buffered.getvalue()).decode('utf-8')
    datums.append({'id': str(idx),
                   'label': labels[image['label']],
                  "img": f'<img src="data:image/jpeg;base64,{b64img}" style="min-width:150px"/>'
                  }
                )

    if idx >= max_embeddings:
        break

output = embed.images(images=images)

embeddings = np.array(output['embeddings'])

atlas.map_data(embeddings=embeddings,
               identifier='cifar',
               data=datums,
               id_field='id',
               topic_model=False)