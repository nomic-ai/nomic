# Map your embeddings
Nomic's neural database ingests unstructured data such as embeddings or text and organizes them.
Once your data is in Atlas, you can view *all* of it at once on a neural map.

## Example maps

[2022 News](https://atlas.nomic.ai/map/357e8f8e-b182-442d-bcbc-a4c3903aeb1b/3c70fef9-1994-4438-92cd-45b0ab803bd7)

[Grand Comics Database](https://atlas.nomic.ai/map/988ad159-0c94-4559-a67a-a0498277b4d8/a5ef1e3c-105f-4606-abdb-5dc2e4fe20af)

[MNIST Logits](https://atlas.nomic.ai/map/2a222eb6-8f5a-405b-9ab8-f5ab23b71cfd/1dae224b-0284-49f7-b7c9-5f80d9ef8b32)

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