from nomic import embed
from nomic import atlas
import numpy as np
import time

long_string = " ".join(["hello" for _ in range(50)])
start = time.time()
output = embed.text(texts=['hello how are you']*50, model='nomic-embed-text-v1')

print(time.time() - start)
print(output['usage'])

embeddings = np.array(output['embeddings'])

print(embeddings.shape)