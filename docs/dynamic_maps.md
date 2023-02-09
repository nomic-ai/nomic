# Dynamic Maps
Maps made in Atlas dynamically update to reflect the underlying data stored in the project.

When you add, update or delete data in an AtlasProject the underlying map records your changes.

!!! note "Project Lock"

    Addition, deletion and update operations a projects data can only occur when the project's transaction lock is released. This lock is present
    when any map is currently building on a project. You can check if this lock is set with the `is_locked` method.

Changes you make to your projects data do not immediately reflect on your map. You must explicitly rebuild your map
to have your changes incorporated into the maps state.

You can commit project data manipulations to your maps state by running the [rebuild_maps](atlas_api.md) method on your AtlasProject.


## Adding data
In the below example, we will create a map and then add data to it. To add data to a project, you should
use the `add_embeddings` or `add_text` methods depending on your project's modality.

The first set of data added to the project will contain 1000 random embeddings in 10 dimensions with mean zero. For
tracking purposes, we will associate and color each embedding with a metadata field called `upload` signifying whether
the embedding was part of the first or second set of data added to the project.

=== "First Upload"

    ``` py
    from nomic import atlas
    import numpy as np
    
    num_embeddings = 1000
    embeddings = np.random.rand(num_embeddings, 10)
    data = [{'upload': 1} for i in range(len(embeddings))]
    
    project = atlas.map_embeddings(embeddings=embeddings,
                                    data=data,
                                    map_name='An Map That Gets Updated',
                                    colorable_fields=['upload'])
    map = project.get_map('An Map That Gets Updated')
    print(map)
    ```

The second upload will contain 1000 random embeddings of dimension 10 but with a different mean. The resulting
maps has two cluster points: one population for the first upload with mean zero and the second
population with the shifted mean.
=== "Second upload"

    ``` py
    embeddings = np.random.rand(num_embeddings, 10) + np.ones(shape=(num_embeddings, 10))
    data = [{'upload': 2} for i in range(len(embeddings))]
    
    project.block_until_accepting_data()
    
    project.add_embeddings(embeddings=embeddings, data=data)
    
    project.rebuild_maps()
    ```

