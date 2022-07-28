from nomic import AtlasClient
from nomic import CohereEmbedder
import numpy as np
from datasets import load_dataset

atlas = AtlasClient()
cohere_api_key = 'zWSuCYVdwA02QbqFudOb3VZxhtfHPTMu1IjeUUoT'

dataset = load_dataset("sentiment140")['train']

max_documents = 1000
subset_idxs = np.random.randint(len(dataset), size=max_documents).tolist()
documents = [dataset[i] for i in subset_idxs]
sentiment_map = {0:'Negative', 1:'Neutral', 2:'Neutral', 3:'Neutral', 4:'Neutral', 5:'Positive'}
for idx, document in enumerate(documents):
    document['id'] = idx
    document['sentiment'] = sentiment_map[document['sentiment']]

embedder = CohereEmbedder(cohere_api_key=cohere_api_key)

print(f"Embedding {len(documents)} documents with Cohere API")
embeddings = embedder.embed(texts=[document['text'] for document in documents],
                            model='medium')

if len(embeddings) != len(documents):
    raise Exception("Embedding job failed")
print("Embedding job complete.")

response = atlas.map_embeddings(embeddings=np.array(embeddings),
                                data=documents,
                                id_field='id',
                                is_public=True,
                                colorable_fields=['sentiment'])

print("Embedding Map:")
print(response)


