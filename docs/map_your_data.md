# Map your embeddings
Atlas ingests unstructured data such as embeddings or text and organizes them.
Once your data is in Atlas, you can view *all of it* at once on an interactive map.


## Your first neural map

The following code snippet shows you how to map your embeddings with Atlas - Nomic's neural database.
Upload 10,000 random embeddings and see them instantly organized on an interactive map.

[Random Embedding Map](https://atlas.nomic.ai/map/74ebf36c-b1fa-4a9e-b091-dcfcc240857e/a9d2e4d0-f5c7-4640-8139-ff858496f45b)

=== "Basic Example"

    ``` py title="map_embeddings.py"
    from nomic import AtlasClient
    import numpy as np
    
    atlas = AtlasClient()
    
    num_embeddings = 10000
    embeddings = np.random.rand(num_embeddings, 256)
    
    response = atlas.map_embeddings(embeddings=embeddings)
    print(response)
    ```

=== "Output"

    ``` bash
    map='https://atlas.nomic.ai/map/74ebf36c-b1fa-4a9e-b091-dcfcc240857e/a9d2e4d0-f5c7-4640-8139-ff858496f45b'
    job_id='ebd5e68e-35b8-4de7-acc4-7abc0c725ca3'
    index_id='66ae0cdf-2d01-440e-9d39-2ef68c3ad445'
    ```

## Colorful Maps

Now let's color add colors. To do this, specify the `data` key in the map call. This field should contain a list
of dictionaries - one for each of your embeddings.

=== "Advanced Example"

    ``` py title="map_embeddings.py"
    from nomic import AtlasClient
    import numpy as np
    
    atlas = AtlasClient()
    
    num_embeddings = 10000
    embeddings = np.random.rand(num_embeddings, 256)
    
    categories = ['rhizome', 'cartography', 'lindenstrauss']
    data = [{'category': categories[i % len(categories)]}
                for i in range(len(embeddings))]
    
    response = atlas.map_embeddings(embeddings=embeddings,
                                    data=data,
                                    colorable_fields=['category']
                                    )
    print(response)
    ```

=== "Output"

    ``` bash
    map='https://atlas.nomic.ai/map/74ebf36c-b1fa-4a9e-b091-dcfcc240857e/a9d2e4d0-f5c7-4640-8139-ff858496f45b'
    job_id='ebd5e68e-35b8-4de7-acc4-7abc0c725ca3'
    index_id='66ae0cdf-2d01-440e-9d39-2ef68c3ad445'
    ```