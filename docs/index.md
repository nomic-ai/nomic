# Atlas
![](assets/atlas_explanation.png)
Meet Atlas - a tool for interacting with both small and internet scale size datasets in your web browser.

Give Atlas unstructured data such as text documents or embedding vectors and Atlas gives you back an interactive web explorer with API access.

Under the hood, Atlas quickly pre-organizes and enriches your data with state-of-the-art AI.

<div align="center">
  <a href="https://colab.research.google.com/drive/1bquOLIaGlu7O_CFc0Wz74HITzWs4UEa4?usp=sharing">Colab Demo</a>
</div>
<div align="center">
  <a href="https://discord.gg/myY5YDR8z8">Discord</a>
</div>


=== "Mapping Embeddings"

    ``` py title="map_embeddings.py"
    from nomic import atlas
    import numpy as np

    num_embeddings = 10000
    embeddings = np.random.rand(num_embeddings, 256)
    
    project = atlas.map_embeddings(embeddings=embeddings)
    print(project.maps)
    ```


Learn how to use Atlas to interact with your data.
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

You are ready to interact with Atlas. Continue on to [make your first map](map_your_data.md).

## Resources:

[Make your first neural map.](map_your_data.md)

[How does Atlas work?](how_does_atlas_work.md)

[Collection of maps.](collection_of_maps.md)

## Example maps

[Twitter](https://atlas.nomic.ai/map/twitter) (5.4 million tweets)

[Stable Diffusion](https://atlas.nomic.ai/map/stablediffusion) (6.4 million images)

[NeurIPS Proceedings](https://atlas.nomic.ai/map/neurips) (16,623 documents)

[ICLR 2018-2023 Submissions](https://atlas.nomic.ai/map/iclr)

[MNIST Logits](https://atlas.nomic.ai/map/2a222eb6-8f5a-405b-9ab8-f5ab23b71cfd/1dae224b-0284-49f7-b7c9-5f80d9ef8b32)



## About us
[Nomic](https://home.nomic.ai) is the world's first *information cartography* company. We believe that the fastest way to understand your
data is to look at it.
