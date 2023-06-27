Atlas allows you to visually and programatically add tags to datapoints. Tags can be added collaboratively by anyone
allowed to edit your project.


You can access and operate on your assigned tags by using the `tags` attribute
of an AtlasMap.


```python
from nomic import AtlasProject

project = AtlasProject(name='My Project')

map = project.maps[0]

map.tags.get_tags()

```




## Atlas Tags
::: nomic.data_operations.AtlasMapTags
    options:
        show_root_heading: True