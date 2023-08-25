Atlas stores your original project data such as text and numeric fields, providing a unified source
for all your dataset's information. These fields are displayed with each point in your Atlas Map.

You can access these uploaded fields programmatically by using the `data` attribute
of an AtlasMap. This is helpful if you would like to perform operations on Atlas artifacts, such as embedding or
topic information, along with your original data.


```python
from nomic import AtlasProject

map = AtlasProject(name='My Project').maps[0]

map.data

```


::: nomic.data_operations.AtlasMapData
    options:
        show_root_heading: True