# Map Your Embeddings
Atlas ingests unstructured data such as embeddings or text and organizes them.
Once your data is in Atlas, you can view *all of it* at once on an interactive map. Any interaction you do on the
map (e.g. tagging, topic labeling, vector search) you can programmatically access in this Python client.


## Your first neural map

The following code snippet shows you how to map your embeddings with Atlas.
Upload 10,000 random embeddings and see them instantly organized on an interactive map.

[Random Embedding Map](https://atlas.nomic.ai/map/82e15baf-5de2-4191-bc60-61ce9d76bd17/91e63b2d-b8af-4de2-a4d2-e6e96d879274)

=== "Basic Example"

    ``` py title="map_embeddings.py"
    from nomic import atlas
    import numpy as np
    
    num_embeddings = 10000
    embeddings = np.random.rand(num_embeddings, 256)
    
    project = atlas.map_embeddings(embeddings=embeddings)
    ```

=== "Output"

    ``` bash
    https://atlas.nomic.ai/map/82e15baf-5de2-4191-bc60-61ce9d76bd17/91e63b2d-b8af-4de2-a4d2-e6e96d879274
    ```

## Add some colors

Now let's add colors. To do this, specify the `data` key in the map call. This field should contain a list
of dictionaries - one for each of your embeddings. In the `map_embeddings` call, specify the key you want to
be able to color by. In our example, this key is `category`.

=== "Advanced Example"

    ``` py title="map_embeddings_with_colors.py"
    from nomic import atlas
    import numpy as np

    num_embeddings = 10000
    embeddings = np.random.rand(num_embeddings, 256)
    
    categories = ['rhizome', 'cartography', 'lindenstrauss']
    data = [{'category': categories[i % len(categories)], 'id': i}
                for i in range(len(embeddings))]
    
    project = atlas.map_embeddings(embeddings=embeddings,
                                    data=data,
                                    id_field='id',
                                    colorable_fields=['category']
                                    )
    ```
