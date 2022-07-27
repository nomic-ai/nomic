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
for idx, document in enumerate(documents):
    document['id'] = idx


embedder = CohereEmbedder(cohere_api_key=cohere_api_key)

print(f"Embedding {len(documents)} documents with Cohere API")
embeddings = embedder.embed(texts=[document['text'] for document in documents],
                            model='small')

print("Embedding job complete.")
print(len(embeddings))

response = atlas.map_embeddings(embeddings=np.array(embeddings),
                                data=documents,
                                id_field='id',
                                is_public=True)

print("Embedding Map:")
print(response)


