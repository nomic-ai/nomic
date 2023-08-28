Atlas groups your data into semantically similar duplicate clusters powered by latent information contained in your embeddings.
Under the hood, Atlas utilizes an algorithm similar to [SemDeDup](https://arxiv.org/abs/2303.09540).


You can access and operate on semantic duplicate clusters programmatically by using the `duplicates` attribute
of an AtlasMap. Make sure to enable duplicate clustering by setting `detect_duplicate = True` when building a map.


```python
from nomic import AtlasProject

map = AtlasProject(name='My Project').maps[0]

map.duplicates

```

::: nomic.data_operations.AtlasMapDuplicates
    options:
        show_root_heading: True