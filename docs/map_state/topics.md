Atlas pre-organizes your data into topics informed by the latent contents of your embeddings. Visually, these are represented by regions of homogenous
color on an Atlas map. 

You can access and operate on topics programmatically by using the `topics` attribute
of an AtlasMap.


```python
from nomic import AtlasProject

map = AtlasProject(name='My Project').maps[0]

map.topics

```


::: nomic.data_operations.AtlasMapTopics
    options:
        show_root_heading: True