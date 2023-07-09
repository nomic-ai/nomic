Atlas stores, manages and generates embeddings for your unstructured data.

You can access Atlas latent embedding (e.g. high dimensional) or their two-dimensional
projected representations.


```python
from nomic import AtlasProject

map = AtlasProject(name='My Project').maps[0]

projected_embeddings = map.embeddings.projected

latent_embeddings = map.embeddings.latent

print(f"The datapoint with id {projected_embeddings['id'][0]} is located at ({projected_embeddings['x'][0]}, {projected_embeddings['y'][0]}) with latent embedding {latent_embeddings[0]}")

```

::: nomic.data_operations.AtlasMapEmbeddings
    options:
        show_root_heading: True