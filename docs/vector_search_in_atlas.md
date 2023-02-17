# Similarity and Vector Search
Atlas supports vector search over maps.

You can think of vector search as a programatic way to access areas of a map.
When you pick a point on a map and use it as input to the `vector_search` function, you get as output points close to
your input point. These outputs are called neighbors.
```python
project = AtlasProject(name='Example Map')
map = project.maps[0]

print(project.get_data(ids=['42']))

neighbors, distances = map.vector_search(ids=['42'])

print(project.get_data(ids=neighbors[0]))
```

## Applications

Vector search can be used to:

1. Programmatically access clusters or neighborhoods of datapoints on your map.
2. Clean your data by finding near duplicates and data points similar to un-wanted ones.
3. Label your data by retrieving points near a point with a known label.
3. Similarity search your data for use-cases like recommendation.


!!! note "Vector Search Operates in the Ambient Space"

    Vector search operates on high dimensional (ambient) vectors corresponding to your data, not on the two dimensional map positions.

## Example
The following example showcases creating a map from 25,000 news articles and then performing a vector search.
First create the map:
```python
from nomic import atlas, AtlasProject
import numpy as np
from datasets import load_dataset
dataset = load_dataset('ag_news')['train']

np.random.seed(0)
max_documents = 25000
subset_idxs = np.random.randint(len(dataset), size=max_documents).tolist()
documents = [dataset[i] for i in subset_idxs]
for idx, document in enumerate(documents):
    document['id'] = idx
    

project = atlas.map_text(data=documents,
                         indexed_field='text',
                         id_field='id',
                         name='News Dataset 25k',
                         colorable_fields=['label'],
                         description='News Dataset 25k'
                          )

```

Then run a vector search:
```python
project = AtlasProject(name='News Dataset 25k')
map = project.maps[0]

#batch two vector search queries into one request.
query_document_ids = [0, 42]
with project.wait_for_project_lock():
    neighbors, distances = map.vector_search(ids=query_document_ids, k=10)

print(neighbors)
data = project.get_data(ids=query_document_ids)
for datum, datum_neighbors in zip(data, neighbors):
    neighbor_data = project.get_data(ids=datum_neighbors)
    print(f"The ten nearest neighbors to the query point {datum} are {neighbor_data}")
```

!!! note "Project Lock and Vector Search"

    You cannot run a vector_search against a map while it's project is locked (e.g. Atlas is building the map).
