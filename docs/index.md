# Atlas
![](assets/atlas_explanation.png)
Meet Atlas - a platform for interacting with both small and internet scale unstructured datasets.
<div align="center">
  <a href="https://colab.research.google.com/drive/1bquOLIaGlu7O_CFc0Wz74HITzWs4UEa4?usp=sharing">Colab Demo</a>
</div>
<div align="center">
  <a href="https://discord.gg/myY5YDR8z8">Discord</a>
</div>

Atlas enables you to:

* Store, update and organize multi-million point datasets of unstructured text, images and embeddings.
* Visually interact with your datasets from a web browser.
* Run semantic search and vector operations over your datasets.

Use Atlas to:

- [Visualize, interact, collaborate and share large datasets of text and embeddings.](map_your_data.md)
- [Collaboratively clean, tag and label your datasets](data_cleaning_in_atlas.md)
- [Build high-availability apps powered by semantic search](https://langchain.readthedocs.io/en/latest/ecosystem/atlas.html)
- [Understand and debug the latent space of your AI model trains](pytorch_embedding_explorer.ipynb)

Read about [how Atlas works](how_does_atlas_work.md) or get started below!

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

You are ready to interact with Atlas. Continue on to [make your first data map](map_your_data.md).

=== "Mapping Embeddings"

    ``` py title="map_embeddings.py"
    from nomic import atlas
    import numpy as np

    num_embeddings = 10000
    embeddings = np.random.rand(num_embeddings, 256)
    
    project = atlas.map_embeddings(embeddings=embeddings)
    print(project.maps)
    ```

## Resources

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
