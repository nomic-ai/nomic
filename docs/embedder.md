# Embedders
You can either embed datums using Atlas' built-in neural inference functionality or supply your own embeddings.
Here is an example of using an external embedding provider API to embed text and then send over the embeddings for
Atlas to organize.


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