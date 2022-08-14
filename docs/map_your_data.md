# Map your data
Nomic's neural database ingests unstructured data such as embeddings or text and organizes them.
Once your data is in Atlas, you can view *all* of it at once on a neural map.

## Example maps

[2022 News](https://atlas.nomic.ai/map/357e8f8e-b182-442d-bcbc-a4c3903aeb1b/3c70fef9-1994-4438-92cd-45b0ab803bd7)

[Grand Comics Database](https://atlas.nomic.ai/map/988ad159-0c94-4559-a67a-a0498277b4d8/a5ef1e3c-105f-4606-abdb-5dc2e4fe20af)

## Your first neural map

The following code snippet shows you how to map your embeddings with Atlas - Nomic's neural database.
Upload 10,000 random embeddings and see them instantly organized on an interactive map.

[Random Embedding Map](https://atlas.nomic.ai/map/74ebf36c-b1fa-4a9e-b091-dcfcc240857e/a9d2e4d0-f5c7-4640-8139-ff858496f45b)

=== "Basic Example"

    ``` py title="map_embeddings.py"
    from nomic import AtlasClient
    import numpy as np
    
    atlas = AtlasClient()
    
    num_embeddings = 10000
    embeddings = np.random.rand(num_embeddings, 256)
    data = [{'id': i} for i in range(len(embeddings))]
    
    response = atlas.map_embeddings(embeddings=embeddings,
                                    data=data,
                                    id_field='id',
                                    is_public=True)
    print(response)
    ```

=== "Output"

    ``` bash
    map='https://atlas.nomic.ai/map/74ebf36c-b1fa-4a9e-b091-dcfcc240857e/a9d2e4d0-f5c7-4640-8139-ff858496f45b'
    job_id='ebd5e68e-35b8-4de7-acc4-7abc0c725ca3'
    index_id='66ae0cdf-2d01-440e-9d39-2ef68c3ad445'
    ```

## Your first neural map of text

Neural maps let you see data through the eyes of a neural network. First we need access to a powerful neural network.
Cohere AI's large language model embedding API will get you started.
Obtain an API key from [cohere.ai](https://os.cohere.ai) to embed your text data.

Add your Cohere API key to the below example to see how their large language model organizes text from a sentiment analysis dataset.

[Sentiment Analysis Map](https://atlas.nomic.ai/map/ff2f89df-451e-49c4-b7a3-a608d7375961/f433cbd1-e728-49da-8c83-685cd613788b)
=== "Co:here Example"

    ``` py title="map_hf_dataset_with_cohere.py"
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
    embeddings = embedder.embed(texts=[document['user'] for document in documents],
                                model='small',
                                num_workers=10,
                                shard_size=1000)
    
    if len(embeddings) != len(documents):
        raise Exception("Embedding job failed")
    print("Embedding job complete.")
    
    response = atlas.map_embeddings(embeddings=np.array(embeddings),
                                    data=documents,
                                    id_field='id',
                                    is_public=True,
                                    colorable_fields=['sentiment'],
                                    num_workers=20)

    print(response)
    

    ```

=== "Output"

    ``` bash
    map='https://atlas.nomic.ai/map/ff2f89df-451e-49c4-b7a3-a608d7375961/f433cbd1-e728-49da-8c83-685cd613788b'
    job_id='b4f97377-e2aa-4305-8bc6-db7f5f6eeabf'
    index_id='46445e68-8c9f-470a-aa82-847e78c0f10e'
    ```