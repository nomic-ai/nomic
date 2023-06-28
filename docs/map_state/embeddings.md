Atlas stores, manages and generates embeddings for your unstructured data.

You can access Atlas embeddings in their ambient form (e.g. high dimensional) or in their two-dimensional
projected representations.


```python
from nomic import AtlasProject

map = AtlasProject(name='My Project').maps[0]

map.embeddings

```

::: nomic.data_operations.AtlasMapEmbeddings
    options:
        show_root_heading: True