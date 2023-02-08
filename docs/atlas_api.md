# Atlas API

This class allows for programmatic interactions with Atlas. Initialize an AtlasProject in any Python context such as a script
or in a Jupyter Notebook to access your web based map interactions.

=== "Atlas Client Example"

    ``` py title="map_embeddings.py"
    from nomic import atlas
    import numpy as np
    
    num_embeddings = 10000
    embeddings = np.random.rand(num_embeddings, 256)
    
    response = atlas.map_embeddings(embeddings=embeddings,
                                    is_public=True)
    print(response)
    ```
=== "Atlas Client Private Map Example"

    ``` py title="map_embeddings_private.py"
    from nomic import atlas
    import numpy as np
    
    num_embeddings = 10000
    embeddings = np.random.rand(num_embeddings, 256)
    
    response = atlas.map_embeddings(embeddings=embeddings,
                                    is_public=False
                                    )
    print(response)
    ```

## Map Embedding API

::: nomic.atlas.map_embeddings
    options:
        show_root_heading: True

## Map Text API
::: nomic.atlas.map_text
    options:
        show_root_heading: True


## AtlasProject API
::: nomic.project.AtlasProject
    options:
        show_root_heading: True


## AtlasProjection API
::: nomic.project.AtlasProjection
    options:
        show_root_heading: True