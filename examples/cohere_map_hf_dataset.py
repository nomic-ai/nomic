from nomic import AtlasClient
from nomic import CohereEmbedder
import numpy as np
from datasets import load_dataset

atlas = AtlasClient()

dataset = load_dataset("sentiment140")['train']

max_documents = 10000
subset_idxs = np.random.randint(len(dataset), size=max_documents).tolist()
documents = [dataset[i] for i in subset_idxs]
for idx, document in enumerate(documents):
    document['id'] = idx

cohere_api_key = ''
embedder = CohereEmbedder(cohere_api_key=cohere_api_key)

print(f"Embedding {len(documents)} documents with Cohere API")
embeddings = embedder.embed(texts=[document['text'] for document in documents],
                            model='small',
                            shard_size=1000,
                            num_workers=10)
print("Embedding job complete.")
print(len(embeddings))

response = atlas.map_embeddings(embeddings=np.array(embeddings), data=documents, unique_id_field='id', is_public=True)

print("Embedding Map:")
print(response)


