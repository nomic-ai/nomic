# Embedders

You need embeddings to make maps. Nomic provides light-weight wrappers on existing embedding providers to quickly
get you going.

=== "Cohere Embedding"

    ``` py title="cohere_embedder_example.py"
    from nomic import CohereEmbedder
    cohere_api_key = ''
    
    embedder = CohereEmbedder(cohere_api_key=cohere_api_key)
    
    print(f"Embedding {len(documents)} documents with Cohere API")
    embeddings = embedder.embed(texts=["Document 1 text", "Document 2 text"],
                                model='small',
                                num_workers=10,
                                shard_size=1000)
    assert len(embeddings) == 2
    ```

::: nomic.CohereEmbedder
    :docstring:
    :members: