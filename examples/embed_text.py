from nomic import embed


embeddings = embed.text(texts=['Hello world'], model='nomic-embed-text-v1')
print(embeddings)