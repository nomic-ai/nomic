# Nomic
[Nomic](https://home.nomic.ai) is the worlds first *information cartography* company. Our first tool, Atlas, allows you to interact with internet scale datasets in your web browser.

=== "Mapping Embeddings"

    ``` py title="map_embeddings.py"
    from nomic import AtlasClient
    import numpy as np
    
    atlas = AtlasClient()
    
    num_embeddings = 10000
    embeddings = np.random.rand(num_embeddings, 256)
    
    response = atlas.map_embeddings(embeddings=embeddings)
    print(response)
    ```

=== "Output"

    ``` bash
    map='https://atlas.nomic.ai/map/74ebf36c-b1fa-4a9e-b091-dcfcc240857e/a9d2e4d0-f5c7-4640-8139-ff858496f45b'
    job_id='ebd5e68e-35b8-4de7-acc4-7abc0c725ca3'
    index_id='66ae0cdf-2d01-440e-9d39-2ef68c3ad445'
    ```

Learn how to use Atlas to interact with your large collections of embeddings and text.

## Quickstart
Install the Nomic client with:
```bash
pip install nomic
```

Login/create your Nomic account:
```bash
nomic login
```

Follow the instructions to obtain your access token. Enter your access token with:
```bash
nomic login [token]
```

You are ready to interact with Nomic services. Continue on to [make your first neural map](map_your_data.md).

## Resources:

[Make your first neural map.](map_your_data.md)

[How does Atlas work?](how_does_atlas_work.md)

[Collection of maps.](collection_of_maps.md)

## Example maps

[Twitter](https://atlas.nomic.ai/map/twitter)

[Stable Diffusion](https://atlas.nomic.ai/map/809ef16a-5b2d-4291-b772-a913f4c8ee61/9ed7d171-650b-4526-85bf-3592ee51ea31) (6.4 Million datums)

[NeurIPS Proceedings](https://atlas.nomic.ai/map/neurips)

[ICLR 2018-2023 Submissions](https://atlas.nomic.ai/map/b06c5cd7-6946-43ed-b515-7934970c8ed7/6e643208-03fb-4b94-ae01-69ce5395ee5b)

[MNIST Logits](https://atlas.nomic.ai/map/2a222eb6-8f5a-405b-9ab8-f5ab23b71cfd/1dae224b-0284-49f7-b7c9-5f80d9ef8b32)



