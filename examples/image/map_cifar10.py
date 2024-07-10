from nomic import atlas
from tqdm import tqdm

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


atlas.map_data(blobs=images,
               data=datums,
               identifier='cifar-50k'
)