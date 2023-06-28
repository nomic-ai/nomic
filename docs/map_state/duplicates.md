Atlas groups your data into semantically similar clusters powered by latent information contained in your embeddings.
Under the hood, it utilizes an algorithm similar to [SemDeDup](https://arxiv.org/abs/2303.09540).


You can access and operate on semantic duplicate clusters programmatically by using the `duplicates` attribute
of an AtlasMap. 


```python
from nomic import AtlasProject

map = AtlasProject(name='My Project').maps[0]

map.duplicates

```

::: nomic.data_operations.AtlasMapDuplicates
    options:
        show_root_heading: True