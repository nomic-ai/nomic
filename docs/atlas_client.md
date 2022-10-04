# Atlas Client

This class allows for programmatic interactions with Atlas - Nomics neural database. Initialize AtlasClient in any Python context such as a script
or in a Jupyter Notebook to organize and interact with your unstructured data.

=== "Atlas Client Example"

    ``` py title="map_embeddings.py"
    from nomic import AtlasClient
    import numpy as np
    
    atlas = AtlasClient()
    
    num_embeddings = 10000
    embeddings = np.random.rand(num_embeddings, 256)
    
    response = atlas.map_embeddings(embeddings=embeddings,
                                    is_public=True)
    print(response)
    ```
=== "Atlas Client Private Map Example"

    ``` py title="map_embeddings_private.py"
    from nomic import AtlasClient
    import numpy as np
    
    atlas = AtlasClient()
    
    num_embeddings = 10000
    embeddings = np.random.rand(num_embeddings, 256)
    
    response = atlas.map_embeddings(embeddings=embeddings,
                                    is_public=False,
                                    organization_name='my_organization'
                                    )
    print(response)
    ```

::: nomic.AtlasClient
    :docstring:
    :members: