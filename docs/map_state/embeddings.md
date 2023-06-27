Atlas stores, manages and generates embeddings for your unstructured data.

You can access Atlas embeddings in their ambient form (e.g. high dimensional) or in their two-dimensional
projected representations.


```python
from nomic import AtlasProject

project = AtlasProject(name='My Project')

map = project.maps[0]

map.embeddings.df

```




## AtlasTopics
::: nomic.data_operations.AtlasMapEmbeddings
    options:
        show_root_heading: True