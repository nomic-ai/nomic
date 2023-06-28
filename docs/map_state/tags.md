Atlas allows you to visually and programatically associate tags to datapoints. Tags can be added collaboratively by anyone
allowed to edit your Atlas Project.


You can access and operate on your assigned tags by using the `tags` attribute
of an AtlasMap.


```python
from nomic import AtlasProject

map = AtlasProject(name='My Project').maps[0]

map.tags

```




## Atlas Tags
::: nomic.data_operations.AtlasMapTags
    options:
        show_root_heading: True