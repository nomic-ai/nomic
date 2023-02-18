# Dynamic Maps
Maps made in Atlas dynamically update to reflect the underlying data stored in the project.

When you add, update or delete data in an AtlasProject the underlying map records your changes.

!!! note "Project Lock"

    Addition, deletion and update operations on a projects data can only occur when the project's transaction lock is released. This lock is present
    when any map is building on a project. You can check if this is set with the `is_locked` property.

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
    data = [{'upload': '1', 'id': i} for i in range(len(embeddings))]
    
    project = atlas.map_embeddings(embeddings=embeddings,
                                   data=data,
                                   id_field='id',
                                   name='A Map That Gets Updated',
                                   colorable_fields=['upload'],
                                   reset_project_if_exists=True)

    map = project.get_map('A Map That Gets Updated')
    print(map)
    ```

The second upload will contain 1000 random embeddings of dimension 10 but with a shifted mean. The resulting
map contains two clusters: one cluster for the first upload with mean zero vectors and the second
cluster corresponding to vectors with the shifted mean.
=== "Second upload"

    ``` py
    total_datums = project.total_datums

    # embeddings with shifted mean.
    embeddings += np.ones(shape=(num_embeddings, 10))
    data = [{'upload': '2', 'id': total_datums+i} for i in range(len(embeddings))]
    
    with project.wait_for_project_lock():
        project.add_embeddings(embeddings=embeddings, data=data)
        project.rebuild_maps()
    ```

!!! note "Project Lock Context Manager"

    Place any logic that needs to wait for a project lock to be released behind the `project.wait_for_project_lock()` context manager.
    This context manager will block the currently running thread until all maps in your project are done building. For large projects,
    this may take a long time.


## Deleting Data
You can delete data with the `delete_data` method on an AtlasProject.

Following the previous example:
=== "Deleting data"

    ``` py
    with project.wait_for_project_lock():
        project.delete_data(ids=[i for i in range(1100, 2000)])
        project.rebuild_maps()
    ```

One cluster in your map should now be 1/10 the size of the other cluster.

