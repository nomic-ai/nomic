"""
This class allows for programmatic interactions with Atlas - Nomic's neural database. Initialize AtlasClient in any Python context such as a script
or in a Jupyter Notebook to organize and interact with your unstructured data.
"""

from typing import Dict, List, Optional

import numpy as np
from loguru import logger
from tqdm import tqdm
import uuid

from .project import AtlasProject
from .settings import *
from .utils import get_random_name

def map_embeddings(
    embeddings: np.array,
    data: List[Dict] = None,
    id_field: str = None,
    name: str = None,
    description: str = None,
    is_public: bool = True,
    colorable_fields: list = [],
    build_topic_model: bool = True,
    topic_label_field: str = None,
    num_workers: None = None,
    organization_name: str = None,
    reset_project_if_exists: bool = False,
    add_datums_if_exists: bool = False,
    shard_size: None = None,
    projection_n_neighbors: int = DEFAULT_PROJECTION_N_NEIGHBORS,
    projection_epochs: int = DEFAULT_PROJECTION_EPOCHS,
    projection_spread: float = DEFAULT_PROJECTION_SPREAD,
) -> AtlasProject:
    '''

    Args:
        embeddings: An [N,d] numpy array containing the batch of N embeddings to add.
        data: An [N,] element list of dictionaries containing metadata for each embedding.
        id_field: Specify your data unique id field. This field can be up 36 characters in length. If not specified, one will be created for you named `id_`.
        name: A name for your map.
        description: A description for your map.
        is_public: Should this embedding map be public? Private maps can only be accessed by members of your organization.
        colorable_fields: The project fields you want to be able to color by on the map. Must be a subset of the projects fields.
        organization_name: The name of the organization to create this project under. You must be a member of the organization with appropriate permissions. If not specified, defaults to your user accounts default organization.
        reset_project_if_exists: If the specified project exists in your organization, reset it by deleting all of its data. This means your uploaded data will not be contextualized with existing data.
        add_datums_if_exists: If specifying an existing project and you want to add data to it, set this to true.
        build_topic_model: Builds a hierarchical topic model over your data to discover patterns.
        topic_label_field: The metadata field to estimate topic labels from. Usually the field you embedded.
        projection_n_neighbors: The number of neighbors to build.
        projection_epochs: The number of epochs to build the map with.
        projection_spread: The spread of the map.

    Returns:
        An AtlasProject that now contains your map.

    '''

    assert isinstance(embeddings, np.ndarray), 'You must pass in a numpy array'

    if embeddings.size == 0:
        raise Exception("Your embeddings cannot be empty")

    if id_field is None:
        id_field = ATLAS_DEFAULT_ID_FIELD

    project_name = get_random_name()
    if description is None:
        description = 'A description for your map.'
    index_name = project_name

    if name:
        project_name = name
        index_name = name
    if description:
        description = description

    if data is None:
        data = [{
            ATLAS_DEFAULT_ID_FIELD: str(uuid.uuid4())
        } for _ in range(len(embeddings))]

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

    # project._validate_map_data_inputs(colorable_fields=colorable_fields, id_field=id_field, data=data)

    number_of_datums_before_upload = project.total_datums

    # sends several requests to allow for threadpool refreshing. Threadpool hogs memory and new ones need to be created.
    logger.info("Uploading embeddings to Atlas.")

    embeddings = embeddings.astype(np.float16)
    if shard_size is not None:
        logger.warning("Passing `shard_size` is deprecated and will raise an error in a future release")
    if num_workers is not None:
        logger.warning("Passing `num_workers` is deprecated and will raise an error in a future release")

    try:
        project.add_embeddings(
            embeddings=embeddings,
            data=data,
        )
    except BaseException as e:
        if number_of_datums_before_upload == 0:
            logger.info(f"{project.name}: Deleting project due to failure in initial upload.")
            project.delete()
        raise e

    logger.info("Embedding upload succeeded.")

    # make a new index if there were no datums in the project before
    if number_of_datums_before_upload == 0:
        projection = project.create_index(
            name=index_name,
            colorable_fields=colorable_fields,
            build_topic_model=build_topic_model,
            projection_n_neighbors=projection_n_neighbors,
            projection_epochs=projection_epochs,
            projection_spread=projection_spread,
            topic_label_field=topic_label_field,
        )
        logger.info(str(projection))
    else:
        # otherwise refresh the maps
        project.rebuild_maps()

    project = project._latest_project_state()
    return project


def map_text(
    data: List[Dict],
    indexed_field: str,
    id_field: str = None,
    name: str = None,
    description: str = None,
    build_topic_model: bool = True,
    multilingual: bool = False,
    is_public: bool = True,
    colorable_fields: list = [],
    num_workers: None = None,
    organization_name: str = None,
    reset_project_if_exists: bool = False,
    add_datums_if_exists: bool = False,
    shard_size: None = None,
    projection_n_neighbors: int = DEFAULT_PROJECTION_N_NEIGHBORS,
    projection_epochs: int = DEFAULT_PROJECTION_EPOCHS,
    projection_spread: float = DEFAULT_PROJECTION_SPREAD,
) -> AtlasProject:
    '''
    Generates or updates a map of the given text.

    Args:
        data: An [N,] element list of dictionaries containing metadata for each embedding.
        indexed_field: The name the data field containing the text your want to map.
        id_field: Specify your data unique id field. This field can be up 36 characters in length. If not specified, one will be created for you named `id_`.
        name: A name for your map.
        description: A description for your map.
        build_topic_model: Builds a hierarchical topic model over your data to discover patterns.
        multilingual: Should the map take language into account? If true, points from different with semantically similar text are considered similar.
        is_public: Should this embedding map be public? Private maps can only be accessed by members of your organization.
        colorable_fields: The project fields you want to be able to color by on the map. Must be a subset of the projects fields.
        organization_name: The name of the organization to create this project under. You must be a member of the organization with appropriate permissions. If not specified, defaults to your user account's default organization.
        reset_project_if_exists: If the specified project exists in your organization, reset it by deleting all of its data. This means your uploaded data will not be contextualized with existing data.
        add_datums_if_exists: If specifying an existing project and you want to add data to it, set this to true.
        projection_n_neighbors: The number of neighbors to build.
        projection_epochs: The number of epochs to build the map with.
        projection_spread: The spread of the map.

    Returns:
        The AtlasProject containing your map.

    '''
    if id_field is None:
        id_field = ATLAS_DEFAULT_ID_FIELD

    project_name = get_random_name()

    if description is None:
        description = 'A description for your map.'
    index_name = project_name

    if name:
        project_name = name
        index_name = name

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
    if shard_size is not None:
        logger.warning("Passing 'shard_size' is deprecated and will be removed in a future release.")
    if num_workers is not None:
        logger.warning("Passing 'num_workers' is deprecated and will be removed in a future release.")
    try:
        project.add_text(
            data,
            shard_size=None,
        )
    except BaseException as e:
        if number_of_datums_before_upload == 0:
            logger.info(f"{project.name}: Deleting project due to failure in initial upload.")
            project.delete()
        raise e

    logger.info("Text upload succeeded.")

    # make a new index if there were no datums in the project before
    if number_of_datums_before_upload == 0:
        projection = project.create_index(
            name=index_name,
            indexed_field=indexed_field,
            colorable_fields=colorable_fields,
            build_topic_model=build_topic_model,
            projection_n_neighbors=projection_n_neighbors,
            projection_epochs=projection_epochs,
            projection_spread=projection_spread,
            multilingual=multilingual,
        )
        logger.info(str(projection))
    else:
        # otherwise refresh the maps
        project.rebuild_maps()

    project = project._latest_project_state()
    return project
