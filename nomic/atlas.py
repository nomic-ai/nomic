"""
This class allows for programmatic interactions with Atlas - Nomic's neural database. Initialize AtlasClient in any Python context such as a script
or in a Jupyter Notebook to organize and interact with your unstructured data.
"""

from typing import Dict, List, Optional

import numpy as np
from loguru import logger
from tqdm import tqdm

from .project import AtlasProject
from .settings import *
from .utils import get_random_name


def map_embeddings(
    embeddings: np.array,
    data: List[Dict] = None,
    id_field: str = None,
    is_public: bool = True,
    colorable_fields: list = [],
    num_workers: int = 10,
    map_name: str = None,
    map_description: str = None,
    organization_name: str = None,
    reset_project_if_exists: bool = False,
    add_datums_if_exists: bool = False,
    shard_size: int = 1000,
    projection_n_neighbors: int = DEFAULT_PROJECTION_N_NEIGHBORS,
    projection_epochs: int = DEFAULT_PROJECTION_EPOCHS,
    projection_spread: float = DEFAULT_PROJECTION_SPREAD,
    build_topic_model: bool = False,
    topic_label_field: str = None,
) -> AtlasProject:
    '''
    Generates a map of the given embeddings.

    **Parameters:**

    * **embeddings** - An [N,d] numpy array containing the batch of N embeddings to add.
    * **data** - An [N,] element list of dictionaries containing metadata for each embedding.
    * **colorable_fields** - The project fields you want to be able to color by on the map. Must be a subset of the projects fields.
    * **id_field** - Specify your datas unique id field. ID fields can be up 36 characters in length. If not specified, one will be created for you named `id_`.
    * **is_public** - Should this embedding map be public? Private maps can only be accessed by members of your organization.
    * **num_workers** - The number of workers to use when sending data.
    * **map_name** - A name for your map.
    * **map_description** - A description for your map.
    * **organization_name** - *(optional)* The name of the organization to create this project under. You must be a member of the organization with appropriate permissions. If not specified, defaults to your user accounts default organization.
    * **reset_project_if_exists** - If the specified project exists in your organization, reset it by deleting all of its data. This means your uploaded data will not be contextualized with existing data.
    * **add_datums_if_exists** - If specifying an existing project and you want to add data to it, set this to true.
    * **shard_size** - The AtlasClient sends your data in shards to Atlas. A smaller shard_size sends more requests. Decrease the shard_size if you hit data size errors during upload.
    * **projection_n_neighbors** - *(optional)* The number of neighbors to use in the projection
    * **projection_epochs** - *(optional)* The number of epochs to use in the projection.
    * **projection_spread** - *(optional)* The effective scale of embedded points. Determines how clumped the map is.
    * **build_topic_model** - Builds a hierarchical topic model over your data to discover patterns.
    * **topic_label_field** - The field to estimate topic labels from.

    **Returns:** A link to your map.
    '''

    if id_field is None:
        id_field = ATLAS_DEFAULT_ID_FIELD

    project_name = get_random_name()
    description = project_name
    index_name = get_random_name()

    if map_name:
        project_name = map_name
        index_name = map_name
    if map_description:
        description = map_description

    if data is None:
        data = [{} for _ in range(len(embeddings))]

    project = AtlasProject(
        name=project_name,
        description=description,
        unique_id_field=id_field,
        modality='embedding',
        is_public=is_public,
        organization_name=organization_name,
        reset_project_if_exists=reset_project_if_exists,
        add_datums_if_exists=add_datums_if_exists,
    )

    project._validate_map_data_inputs(colorable_fields=colorable_fields, id_field=id_field, data=data)

    number_of_datums_before_upload = project.total_datums

    # sends several requests to allow for threadpool refreshing. Threadpool hogs memory and new ones need to be created.
    logger.info("Uploading embeddings to Atlas.")

    embeddings = embeddings.astype(np.float16)
    with tqdm(total=len(data) // shard_size) as pbar:
        for i in range(0, len(data), MAX_MEMORY_CHUNK):
            try:
                project.add_embeddings(
                    embeddings=embeddings[i : i + MAX_MEMORY_CHUNK, :],
                    data=data[i : i + MAX_MEMORY_CHUNK],
                    shard_size=shard_size,
                    num_workers=num_workers,
                    pbar=pbar,
                )
            except BaseException as e:
                if number_of_datums_before_upload == 0:
                    logger.info("Deleting project due to failure in initial upload.")
                    project.delete()
                raise e

    logger.info("Embedding upload succeeded.")

    # make a new index if there were no datums in the project before
    if number_of_datums_before_upload == 0:
        create_index_response = project.create_index(
            index_name=index_name,
            colorable_fields=colorable_fields,
            build_topic_model=build_topic_model,
            projection_n_neighbors=projection_n_neighbors,
            projection_epochs=projection_epochs,
            projection_spread=projection_spread,
            topic_label_field=topic_label_field,
        )
    else:
        # otherwise refresh the maps
        project.refresh_maps()
        return project

    project = project._latest_project_state()
    logger.info(dict(create_index_response))
    return project


def map_text(
    data: List[Dict],
    indexed_field: str,
    id_field: str = None,
    is_public: bool = True,
    colorable_fields: list = [],
    num_workers: int = 10,
    map_name: str = None,
    map_description: str = None,
    organization_name: str = None,
    reset_project_if_exists: bool = False,
    add_datums_if_exists: bool = False,
    shard_size: int = 1000,
    projection_n_neighbors: int = DEFAULT_PROJECTION_N_NEIGHBORS,
    projection_epochs: int = DEFAULT_PROJECTION_EPOCHS,
    projection_spread: float = DEFAULT_PROJECTION_SPREAD,
    build_topic_model: bool = False,
    multilingual: bool = False,
):
    '''
    Generates or updates a map of the given text.

    **Parameters:**

    * **data** - An [N,] element list of dictionaries containing metadata for each embedding.
    * **indexed_field** - The name the data field corresponding to the text to be mapped.
    * **id_field** - Specify your datas unique id field. ID fields can be up 36 characters in length. If not specified, one will be created for you named `id_`.
    * **colorable_fields** - The project fields you want to be able to color by on the map. Must be a subset of the projects fields.
    * **is_public** - Should this embedding map be public? Private maps can only be accessed by members of your organization.
    * **num_workers** - The number of workers to use when sending data.
    * **map_name** - A name for your map.
    * **map_description** - A description for your map.
    * **organization_name** - *(optional)* The name of the organization to create this project under. You must be a member of the organization with appropriate permissions. If not specified, defaults to your user accounts default organization.
    * **reset_project_if_exists** - If the specified project exists in your organization, reset it by deleting all of its data. This means your uploaded data will not be contextualized with existing data.
    * **add_datums_if_exists** - If specifying an existing project and you want to add data to it, set this to true.        * **shard_size** - The AtlasClient sends your data in shards to Atlas. A smaller shard_size sends more requests. Decrease the shard_size if you hit data size errors during upload.
    * **projection_n_neighbors** - *(optional)* The number of neighbors to use in the projection
    * **projection_epochs** - *(optional)* The number of epochs to use in the projection.
    * **projection_spread** - *(optional)* The effective scale of embedded points. Determines how clumped the map is.
    * **build_topic_model** - Builds a hierarchical topic model over your data to discover patterns.
    * **multilingual** - Should the map take language into account? If true, points from different languages but semantically similar text are close together.

    **Returns:** A link to your map.
    '''
    if id_field is None:
        id_field = ATLAS_DEFAULT_ID_FIELD

    project_name = get_random_name()
    description = project_name
    index_name = get_random_name()

    if map_name:
        project_name = map_name
        index_name = map_name
    if map_description:
        description = map_description

    project = AtlasProject(
        name=project_name,
        description=description,
        unique_id_field=id_field,
        modality='text',
        is_public=is_public,
        organization_name=organization_name,
        reset_project_if_exists=reset_project_if_exists,
        add_datums_if_exists=add_datums_if_exists,
    )

    project._validate_map_data_inputs(colorable_fields=colorable_fields, id_field=id_field, data=data)

    number_of_datums_before_upload = project.total_datums

    logger.info("Uploading text to Atlas.")

    with tqdm(total=len(data) // shard_size) as pbar:
        for i in range(0, len(data), MAX_MEMORY_CHUNK):
            try:
                project.add_text(
                    data=data[i : i + MAX_MEMORY_CHUNK],
                    shard_size=shard_size,
                    num_workers=num_workers,
                    pbar=pbar,
                )
            except BaseException as e:
                if number_of_datums_before_upload == 0:
                    logger.info("Deleting project due to failure in initial upload.")
                    project.delete()
                raise e

    logger.info("Text upload succeeded.")

    # make a new index if there were no datums in the project before
    if number_of_datums_before_upload == 0:
        response = project.create_index(
            index_name=index_name,
            indexed_field=indexed_field,
            colorable_fields=colorable_fields,
            build_topic_model=build_topic_model,
            projection_n_neighbors=projection_n_neighbors,
            projection_epochs=projection_epochs,
            projection_spread=projection_spread,
            multilingual=multilingual,
        )
        return dict(response)
    else:
        # otherwise refresh the maps
        project.refresh_maps()
        return project

    logger.info(dict(response))

    return project
