from nomic import AtlasClient
from nomic import CohereEmbedder
import numpy as np
from datasets import load_dataset

atlas = AtlasClient()
cohere_api_key = ''

dataset = load_dataset("sentiment140")['train']

max_documents = 10000
subset_idxs = np.random.randint(len(dataset), size=max_documents).tolist()
documents = [dataset[i] for i in subset_idxs]

embedder = CohereEmbedder(cohere_api_key=cohere_api_key)

print(f"Embedding {len(documents)} documents with Cohere API")
embeddings = embedder.embed(texts=[document['user'] for document in documents],
                            model='small',
                            num_workers=10,
                            shard_size=1000,)

if len(embeddings) != len(documents):
    raise Exception("Embedding job failed")
print("Embedding job complete.")

response = atlas.map_embeddings(embeddings=np.array(embeddings),
                                data=documents,
                                colorable_fields=['sentiment'],
                                is_public=True,
                                map_name='Sentiment 140',
                                map_description='A 10,000 point sample of the huggingface sentiment140 dataset embedded with the co:here small model.',
                                organization_name=None, #defaults to your current user.
                                num_workers=20)
print(response)
