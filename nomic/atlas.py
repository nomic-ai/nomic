"""
This class allows for programmatic interactions with Atlas - Nomic's neural database. Initialize AtlasClient in any Python context such as a script
or in a Jupyter Notebook to organize and interact with your unstructured data.
"""

import uuid
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pyarrow as pa
from loguru import logger
from pandas import DataFrame
from tqdm import tqdm

from .data_inference import NomicDuplicatesOptions, NomicEmbedOptions, NomicProjectOptions, NomicTopicOptions
from .dataset import AtlasDataset
from .settings import *
from .utils import arrow_iterator, b64int, get_random_name


def map_data(
    data: Union[DataFrame, List[Dict], pa.Table, None] = None,
    embeddings: np.array = None,
    name: str = None,
    description: str = "",
    id_field: str = None,
    is_public: bool = True,
    indexed_field: str = None,
    projection: Union[bool, Dict, NomicProjectOptions] = True,
    topic_model: Union[bool, Dict, NomicTopicOptions] = True,
    duplicate_detection: Union[bool, Dict, NomicDuplicatesOptions] = True,
    embedding_model: Optional[Union[str, Dict, NomicEmbedOptions]] = None,
) -> AtlasDataset:
    """

    Args:
        data: An ordered collection of the datapoints you are structuring. Can be a list of dictionaries, Pandas Dataframe or PyArrow Table.
        embeddings: An [N,d] numpy array containing the N embeddings to add.
        name: The name of your dataset
        description: The description of your dataset
        id_field: Specify your data unique id field. This field can be up 36 characters in length. If not specified, one will be created for you named `id_`.
        is_public: Should the dataset be accessible outside your Nomic Atlas organization.
        projection: Options to adjust Nomic Project - the dimensionality algorithm organizing your dataset.
        topic_model: Options to adjust Nomic Topic - the topic model organizing your dataset.
        duplicate_detection: Options to adjust Nomic Duplicates - the duplicate detection algorithm.
        embedding_model: Options to adjust the embedding model used to embed your dataset.
    :return:
    """
    if embeddings is not None:
        modality = 'embedding'
        assert isinstance(embeddings, np.ndarray), 'You must pass in a numpy array'
        if embeddings.size == 0:
            raise Exception("Your embeddings cannot be empty")

    if indexed_field is not None:
        modality = 'text'

    if id_field is None:
        id_field = ATLAS_DEFAULT_ID_FIELD

    project_name = get_random_name()

    index_name = project_name

    if name:
        project_name = name
        index_name = name
    if description:
        description = description

    # no metadata was specified
    added_id_field = False
    if data is None:
        data = [{ATLAS_DEFAULT_ID_FIELD: b64int(i)} for i in range(len(embeddings))]
        added_id_field = True

    if id_field == ATLAS_DEFAULT_ID_FIELD and id_field not in data[0]:
        added_id_field = True
        for i in range(len(data)):
            # do not modify object the user passed in - also ensures IDs are unique if two input datums are the same *object*
            data[i] = data[i].copy()
            data[i][id_field] = b64int(i)

    if added_id_field:
        logger.warning("An ID field was not specified in your data so one was generated for you in insertion order.")

    dataset = AtlasDataset(
        identifier=project_name, description=description, unique_id_field=id_field, is_public=is_public
    )

    number_of_datums_before_upload = dataset.total_datums

    if number_of_datums_before_upload > 0:
        raise Exception('Cannot use map_data to update an existing dataset.')

    # Add data by modality
    logger.info("Uploading data to Atlas.")
    try:
        if modality == 'text':
            dataset.add_data(data=data)
        elif modality == 'embedding':
            dataset.add_data(
                embeddings=embeddings,
                data=data,
            )
    except BaseException as e:
        if number_of_datums_before_upload == 0:
            logger.info(f"{dataset.identifier}: Deleting dataset due to failure in initial upload.")
            dataset.delete()
        raise e

    logger.info(f"`{dataset.identifier}`: Data upload succeeded to dataset`")

    projection = dataset.create_index(
        name=index_name,
        indexed_field=indexed_field,
        modality=modality,
        projection=projection,
        topic_model=topic_model,
        duplicate_detection=duplicate_detection,
        embedding_model=embedding_model,
    )

    project = dataset._latest_project_state()
    return project


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
    reset_project_if_exists: bool = False,
    add_datums_if_exists: bool = False,
    shard_size: None = None,
    projection_n_neighbors: int = DEFAULT_PROJECTION_N_NEIGHBORS,
    projection_epochs: int = DEFAULT_PROJECTION_EPOCHS,
    projection_spread: float = DEFAULT_PROJECTION_SPREAD,
    organization_name=None,
) -> AtlasDataset:
    '''

    Args:
        embeddings: An [N,d] numpy array containing the batch of N embeddings to add.
        data: An [N,] element list of dictionaries containing metadata for each embedding.
        id_field: Specify your data unique id field. This field can be up 36 characters in length. If not specified, one will be created for you named `id_`.
        name: A name for your dataset. Specify in the format `organization/project` to create in a specific organization.
        description: A description for your map.
        is_public: Should this embedding map be public? Private maps can only be accessed by members of your organization.
        colorable_fields: The dataset fields you want to be able to color by on the map. Must be a subset of the projects fields.
        reset_project_if_exists: If the specified dataset exists in your organization, reset it by deleting all of its data. This means your uploaded data will not be contextualized with existing data.
        add_datums_if_exists: If specifying an existing dataset and you want to add data to it, set this to true.
        build_topic_model: Builds a hierarchical topic model over your data to discover patterns.
        topic_label_field: The metadata field to estimate topic labels from. Usually the field you embedded.
        projection_n_neighbors: The number of neighbors to build.
        projection_epochs: The number of epochs to build the map with.
        projection_spread: The spread of the map.

    Returns:
        An AtlasDataset that now contains your map.

    '''

    assert isinstance(embeddings, np.ndarray), 'You must pass in a numpy array'

    raise DeprecationWarning("map_embeddings is deprecated and will soon be removed, use atlas.map_data instead.")

    if organization_name is not None:
        logger.warning(
            "Passing organization name has been removed in Nomic Python client 3.0. Instead identify your dataset with `organization_name/project_name` (e.g. sterling-cooper/november-ads)."
        )

    logger.warning("map_embeddings is deprecated and will soon be removed, you should use map_data instead.")
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

    added_id_field = False
    if data is None:
        data = [{ATLAS_DEFAULT_ID_FIELD: b64int(i)} for i in range(len(embeddings))]
        added_id_field = True

    if id_field == ATLAS_DEFAULT_ID_FIELD and id_field not in data[0]:
        added_id_field = True
        for i in range(len(data)):
            # do not modify object the user passed in - also ensures IDs are unique if two input datums are the same *object*
            data[i] = data[i].copy()
            data[i][id_field] = b64int(i)

    if added_id_field:
        logger.warning("An ID field was not specified in your data so one was generated for you in insertion order.")

    project = AtlasDataset(
        identifier=project_name,
        description=description,
        unique_id_field=id_field,
        is_public=is_public,
        reset_project_if_exists=reset_project_if_exists,
        add_datums_if_exists=add_datums_if_exists,
    )

    number_of_datums_before_upload = project.total_datums

    # sends several requests to allow for threadpool refreshing. Threadpool hogs memory and new ones need to be created.
    logger.info("Uploading embeddings to Atlas.")

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
            logger.info(f"{project.identifier}: Deleting dataset due to failure in initial upload.")
            project.delete()
        raise e

    logger.info("Embedding upload succeeded.")

    # make a new index if there were no datums in the dataset before
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
    else:
        # otherwise refresh the maps
        project.rebuild_maps()

    project = project._latest_project_state()
    return project


def map_text(
    data: Union[Iterable[Dict], DataFrame],
    indexed_field: str,
    id_field: str = None,
    name: str = None,
    description: str = None,
    build_topic_model: bool = True,
    is_public: bool = True,
    colorable_fields: list = [],
    num_workers: None = None,
    reset_project_if_exists: bool = False,
    add_datums_if_exists: bool = False,
    shard_size: None = None,
    projection_n_neighbors: int = DEFAULT_PROJECTION_N_NEIGHBORS,
    projection_epochs: int = DEFAULT_PROJECTION_EPOCHS,
    projection_spread: float = DEFAULT_PROJECTION_SPREAD,
    duplicate_detection: bool = True,
    duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
    organization_name=None,
) -> AtlasDataset:
    '''
    Generates or updates a map of the given text.

    Args:
        data: An [N,] element iterable of dictionaries containing metadata for each embedding.
        indexed_field: The name the data field containing the text your want to map.
        id_field: Specify your data unique id field. This field can be up 36 characters in length. If not specified, one will be created for you named `id_`.
        name: A name for your dataset. Specify in the format `organization/project` to create in a specific organization.
        description: A description for your map.
        build_topic_model: Builds a hierarchical topic model over your data to discover patterns.
        is_public: Should this embedding map be public? Private maps can only be accessed by members of your organization.
        colorable_fields: The dataset fields you want to be able to color by on the map. Must be a subset of the projects fields.
        reset_project_if_exists: If the specified dataset exists in your organization, reset it by deleting all of its data. This means your uploaded data will not be contextualized with existing data.
        add_datums_if_exists: If specifying an existing dataset and you want to add data to it, set this to true.
        projection_n_neighbors: The number of neighbors to build.
        projection_epochs: The number of epochs to build the map with.
        projection_spread: The spread of the map.

    Returns:
        The AtlasDataset containing your map.

    '''
    if organization_name is not None:
        logger.warning(
            "Passing organization name has been removed in Nomic Python client 3.0. Instead identify your dataset with `organization_name/project_name` (e.g. sterling-cooper/november-ads)."
        )

    raise DeprecationWarning("map_text is deprecated and will soon be removed, use atlas.map_data instead.")

    logger.warning("map_text is deprecated and will soon be removed, you should use map_data instead.")

    if id_field is None:
        id_field = ATLAS_DEFAULT_ID_FIELD

    project_name = get_random_name()

    if description is None:
        description = 'A description for your map.'
    index_name = project_name

    if name:
        project_name = name
        index_name = name

    if isinstance(data, DataFrame):
        # Convert DataFrame to a generator of dictionaries
        data_iterator = (row._asdict() for row in data.itertuples(index=False))
    elif isinstance(data, pa.Table):
        # Create generator from pyarrow table
        data_iterator = arrow_iterator(data)
    else:
        data_iterator = iter(data)

    first_sample = next(data_iterator, None)

    if first_sample is None:
        logger.warning("Passed data has no samples. No dataset will be created")
        return

    project = AtlasDataset(
        identifier=project_name,
        description=description,
        unique_id_field=id_field,
        is_public=is_public,
        reset_project_if_exists=reset_project_if_exists,
        add_datums_if_exists=add_datums_if_exists,
    )

    add_id_field = False

    if id_field == ATLAS_DEFAULT_ID_FIELD and id_field not in first_sample:
        add_id_field = True

    if add_id_field:
        logger.warning("An ID field was not specified in your data so one was generated for you in insertion order.")

    project._validate_map_data_inputs(colorable_fields=colorable_fields, id_field=id_field, data_sample=first_sample)

    number_of_datums_before_upload = project.total_datums

    logger.info("Uploading text to Atlas.")
    if shard_size is not None:
        logger.warning("Passing 'shard_size' is deprecated and will be removed in a future release.")
    if num_workers is not None:
        logger.warning("Passing 'num_workers' is deprecated and will be removed in a future release.")
    try:
        upload_batch_size = 100_000
        id_to_add = 0
        if add_id_field:
            first_sample = first_sample.copy()
            first_sample[id_field] = b64int(id_to_add)
            id_to_add += 1

        batch = [first_sample]

        for d in data_iterator:
            if add_id_field:
                # do not modify object the user passed in - also ensures IDs are unique if two input datums are the same *object*
                d = d.copy()
                # necessary to persist change
                d[id_field] = b64int(id_to_add)
                id_to_add += 1
            batch.append(d)
            if len(batch) >= upload_batch_size:
                project.add_text(batch)
                batch = []

        if len(batch) > 0:
            project.add_text(batch)

    except BaseException as e:
        if number_of_datums_before_upload == 0:
            logger.info(f"{project.name}: Deleting dataset due to failure in initial upload.")
            project.delete()
        raise e

    logger.info("Text upload succeeded.")

    # make a new index if there were no datums in the dataset before
    if number_of_datums_before_upload == 0:
        projection = project.create_index(
            name=index_name,
            indexed_field=indexed_field,
            colorable_fields=colorable_fields,
            build_topic_model=build_topic_model,
            projection_n_neighbors=projection_n_neighbors,
            projection_epochs=projection_epochs,
            projection_spread=projection_spread,
            duplicate_detection=duplicate_detection,
            duplicate_threshold=duplicate_threshold,
        )
        logger.info(str(projection))
    else:
        # otherwise refresh the maps
        project.rebuild_maps()

    project = project._latest_project_state()
    return project
