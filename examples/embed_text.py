from nomic import embeddings


embeddings = embeddings.text(texts=['Hello world'], model='nomic-embed-text-v1')
print(embeddings)