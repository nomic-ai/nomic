# Updating Maps
To update a map with new data, simply send the new data to the project inside a map call.
When you upload new data to a project, every index in that project will be updated with that new data.
A minimal example of this for text projects is shown below:

=== "Progressive Upload Example"

    ``` py title="progressive_upload.py"
    from nomic import AtlasClient
    import numpy as np
    import time
    from datasets import load_dataset
    
    atlas = AtlasClient()
    
    num_embeddings = 1000
    embeddings = np.random.rand(num_embeddings, 10)
    data = [{'upload': 0.0} for i in range(len(embeddings))]
    
    response = atlas.map_embeddings(embeddings=embeddings,
                                    map_name='An Map That Gets Updated',
                                    data=data,
                                    colorable_fields=['upload'],
                                    is_public=True,
                                    reset_project_if_exists=True)
    
    #second upload of data, shift the mean of the embeddings
    embeddings = np.random.rand(num_embeddings, 10) + np.ones(shape=(num_embeddings, 10))
    data = [{'upload': 1.0} for i in range(len(embeddings))]
    
    current_project = atlas.get_project(response['project_name'])
    
    while True:
        time.sleep(10)
        if atlas.is_project_accepting_data(project_id=current_project['id']):
            response = atlas.map_embeddings(embeddings=embeddings,
                                            map_name=current_project['project_name'],
                                            colorable_fields=['upload'],
                                            data=data,
                                            )
            break
    ```