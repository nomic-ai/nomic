from nomic import atlas, embed
import numpy as np
from datasets import load_dataset

dataset = load_dataset('ag_news')['train']

max_documents = 10000
subset_idxs = np.random.choice(len(dataset), size=max_documents, replace=True).tolist()
documents = [dataset[i] for i in subset_idxs]


def generate_embeddings(documents):
    batch_size = 250
    document_embeddings = []

    batch = []
    for idx, doc in enumerate(documents):
        batch.append(doc['text'])
        if (idx + 1) % batch_size == 0:
            batch_embeddings = embed.text(texts=batch, model='nomic-embed-text-v1')['embeddings']
            for item in batch_embeddings:
                document_embeddings.append(item)
            print(idx)
            batch = []
    document_embeddings = np.array(document_embeddings)
    return document_embeddings

document_embeddings = generate_embeddings(documents)

print(document_embeddings.shape)
response = atlas.map_embeddings(embeddings=document_embeddings, data=documents, topic_label_field='text', build_topic_model=True)
print(response)