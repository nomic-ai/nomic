from nomic import embed
import base64
from io import BytesIO
import numpy as np

from datasets import load_dataset

dataset = load_dataset("mnist")['train']



images = []
for idx, image in enumerate(dataset):
    print(image)
    images.append(image['image'])
    if idx == 10:
        break

embeddings = embed.images(images=images)

print(embeddings)
print(np.array(embeddings).shape)


