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
from PIL import Image
from pyarrow import Table
from tqdm import tqdm

from .data_inference import NomicDuplicatesOptions, NomicEmbedOptions, NomicProjectOptions, NomicTopicOptions
from .dataset import AtlasDataset, AtlasDataStream
from .settings import *
from .utils import arrow_iterator, b64int, get_random_name


def map_data(
    data: Optional[Union[DataFrame, List[Dict], Table]] = None,
    blobs: Optional[List[Union[str, bytes, Image.Image]]] = None,
    embeddings: Optional[np.ndarray] = None,
    identifier: Optional[str] = None,
    description: str = "",
    id_field: Optional[str] = None,
    is_public: bool = True,
    indexed_field: Optional[str] = None,
    projection: Union[bool, Dict, NomicProjectOptions] = True,
    topic_model: Union[bool, Dict, NomicTopicOptions] = True,
    duplicate_detection: Union[bool, Dict, NomicDuplicatesOptions] = True,
    embedding_model: Optional[Union[str, Dict, NomicEmbedOptions]] = None,
) -> AtlasDataset:
    """

    Args:
        data: An ordered collection of the datapoints you are structuring. Can be a list of dictionaries, Pandas Dataframe or PyArrow Table.
        blobs: A list of image paths, bytes, or PIL images to add to your image dataset that are stored locally.
        embeddings: An [N,d] numpy array containing the N embeddings to add.
        identifier: A name for your dataset that is used to generate the dataset identifier. A unique name will be chosen if not supplied.
        description: The description of your dataset
        id_field: Specify your data unique id field. This field can be up 36 characters in length. If not specified, one will be created for you named `id_`.
        is_public: Should the dataset be accessible outside your Nomic Atlas organization.
        projection: Options to adjust Nomic Project - the dimensionality algorithm organizing your dataset.
        topic_model: Options to adjust Nomic Topic - the topic model organizing your dataset.
        duplicate_detection: Options to adjust Nomic Duplicates - the duplicate detection algorithm.
        embedding_model: Options to adjust the embedding model used to embed your dataset.
    :return:
    """
    modality = "embedding"
    if embeddings is not None:
        assert isinstance(embeddings, np.ndarray), "You must pass in a numpy array"
        if embeddings.size == 0:
            raise Exception("Your embeddings cannot be empty")

    if indexed_field is not None:
        if embeddings is not None:
            logger.warning("You have specified an indexed field but are using embeddings. Embeddings will be ignored.")
        modality = "text"

    if blobs is not None:
        # change this when we support other modalities
        modality = "image"
        indexed_field = "_blob_hash"
        if embedding_model is not None:
            if isinstance(embedding_model, str):
                model_name = embedding_model
            elif isinstance(embedding_model, dict):
                model_name = embedding_model["model"]
            elif isinstance(embedding_model, NomicEmbedOptions):
                model_name = embedding_model.model
            else:
                raise ValueError("embedding_model must be a string, dictionary, or NomicEmbedOptions object")

            if model_name in ["nomic-embed-text-v1", "nomic-embed-text-v1.5"]:
                raise Exception("You cannot use a text embedding model with blobs")
        else:
            # default to vision v1.5
            embedding_model = NomicEmbedOptions(model="nomic-embed-vision-v1.5")

    if id_field is None:
        id_field = ATLAS_DEFAULT_ID_FIELD

    project_name = get_random_name()

    dataset_name = project_name
    index_name = dataset_name

    if identifier:
        dataset_name = identifier
        index_name = identifier
    if description:
        description = description

    # no metadata was specified
    added_id_field = False

    if data is None:
        added_id_field = True
        if embeddings is not None:
            data = [{ATLAS_DEFAULT_ID_FIELD: b64int(i)} for i in range(len(embeddings))]
        elif blobs is not None:
            data = [{ATLAS_DEFAULT_ID_FIELD: b64int(i)} for i in range(len(blobs))]
        else:
            raise ValueError("You must specify either data, embeddings, or blobs")

    if id_field == ATLAS_DEFAULT_ID_FIELD and data is not None:
        if isinstance(data, list) and id_field not in data[0]:
            added_id_field = True
            for i in range(len(data)):
                # do not modify object the user passed in - also ensures IDs are unique if two input datums are the same *object*
                data[i] = data[i].copy()
                data[i][id_field] = b64int(i)
        elif isinstance(data, DataFrame) and id_field not in data.columns:
            data[id_field] = [b64int(i) for i in range(data.shape[0])]
            added_id_field = True
        elif isinstance(data, pa.Table) and not id_field in data.column_names:  # type: ignore
            ids = pa.array([b64int(i) for i in range(len(data))])
            data = data.append_column(id_field, ids)  # type: ignore
            added_id_field = True
        elif id_field not in data[0]:
            raise ValueError("map_data data must be a list of dicts, a pandas dataframe, or a pyarrow table")

    if added_id_field:
        logger.warning("An ID field was not specified in your data so one was generated for you in insertion order.")

    dataset = AtlasDataset(
        identifier=dataset_name, description=description, unique_id_field=id_field, is_public=is_public
    )

    number_of_datums_before_upload = dataset.total_datums

    if number_of_datums_before_upload > 0:
        raise Exception("Cannot use map_data to update an existing dataset.")

    # Add data by modality
    logger.info("Uploading data to Atlas.")
    try:
        if modality == "text":
            dataset.add_data(data=data)
        elif modality == "embedding":
            dataset.add_data(
                embeddings=embeddings,
                data=data,
            )
        elif modality == "image":
            dataset.add_data(blobs=blobs, data=data)

    except BaseException as e:
        if number_of_datums_before_upload == 0:
            logger.info(f"{dataset.identifier}: Deleting dataset due to failure in initial upload.")
            dataset.delete()
        raise e

    logger.info(f"`{dataset.identifier}`: Data upload succeeded to dataset`")

    dataset.create_index(
        name=index_name,
        indexed_field=indexed_field,
        modality=modality,
        projection=projection,
        topic_model=topic_model,
        duplicate_detection=duplicate_detection,
        embedding_model=embedding_model,
    )

    dataset = dataset._latest_dataset_state()
    return dataset


def map_embeddings(
    embeddings: np.ndarray,
    data: Optional[List[Dict]] = None,
    id_field: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    is_public: bool = True,
    colorable_fields: list = [],
    build_topic_model: bool = True,
    topic_label_field: Optional[str] = None,
    num_workers: None = None,
    reset_project_if_exists: bool = False,
    add_datums_if_exists: bool = False,
    shard_size: None = None,
    projection_n_neighbors: int = DEFAULT_PROJECTION_N_NEIGHBORS,
    projection_epochs: int = DEFAULT_PROJECTION_EPOCHS,
    projection_spread: float = DEFAULT_PROJECTION_SPREAD,
    organization_name=None,
) -> AtlasDataset:
    """

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

    """

    assert isinstance(embeddings, np.ndarray), "You must pass in a numpy array"
    raise DeprecationWarning("map_embeddings is deprecated and will soon be removed, use atlas.map_data instead.")


def map_text(
    data: Union[Iterable[Dict], DataFrame],
    indexed_field: str,
    id_field: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
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
    """
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

    """
    if organization_name is not None:
        logger.warning(
            "Passing organization name has been removed in Nomic Python client 3.0. Instead identify your dataset with `organization_name/project_name` (e.g. sterling-cooper/november-ads)."
        )
    raise DeprecationWarning("map_text is deprecated and will soon be removed, use atlas.map_data instead.")


# NOTE: This will be deprecated for AtlasDataStream class
def _get_datastream_credentials(name: Optional[str] = "contrastors") -> Dict[str, str]:
    """
    Returns credentials for a datastream.

    Args:
        name: Datastream name
    Returns:
        A dictionary with credentials to access a datastream.
    """
    atlas_data_access = AtlasDataStream(name)
    return atlas_data_access.get_credentials()
