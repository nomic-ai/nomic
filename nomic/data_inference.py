from typing import Any, Dict, List, Literal, Optional, Union

import pyarrow as pa
from pydantic import AliasChoices, BaseModel, Field

from .settings import DEFAULT_DUPLICATE_THRESHOLD


def from_list(values: Dict[str, Any], schema=None) -> pa.Table:
    tb = pa.Table.from_pylist(values, schema=schema)
    return tb


permitted_types = {
    "integer": pa.int32(),
    "float": pa.float32(),
    "date": pa.timestamp("ms"),
    "string": pa.string(),
    "categorical": pa.string(),
}


def convert_pyarrow_schema_for_atlas(schema: pa.Schema) -> pa.Schema:
    """
    Convert a pyarrow schema to one with types that match the subset of types supported by Atlas for upload.
    """
    types = {}
    whitelist = {}
    for field in schema:
        if field.name.startswith("_"):
            # Underscore fields are private to Atlas and will be handled with their own logic.
            if not field.name in {"_embeddings", "_blob_hash"}:
                raise ValueError(f"Underscore fields are reserved for Atlas internal use: {field.name}")
            whitelist[field.name] = field.type
        elif pa.types.is_boolean(field.type):
            raise TypeError(f"Boolean type not supported: {field.name}")
        elif pa.types.is_list(field.type):
            raise TypeError(f"List types not supported: {field.name}")
        elif pa.types.is_struct(field.type):
            raise TypeError(f"Struct types not supported: {field.name}")
        elif pa.types.is_dictionary(field.type):
            types[field.name] = "categorical"
        elif pa.types.is_string(field.type):
            types[field.name] = "string"
        elif pa.types.is_integer(field.type):
            types[field.name] = "integer"
        elif pa.types.is_floating(field.type):
            types[field.name] = "float"
        elif pa.types.is_timestamp(field.type):
            types[field.name] = "date"
        elif pa.types.is_temporal(field.type):
            types[field.name] = "date"
        else:
            raise TypeError(f"Unknown type: {field.name} {field.type}")
    usertypes = {k: permitted_types[v] for k, v in types.items()}

    return pa.schema({**usertypes, **whitelist})


class NomicProjectOptions(BaseModel):
    """
    Options for Nomic 2D Dimensionality Reduction Model

    Args:
        n_neighbors: The number of neighbors to use when approximating the high dimensional embedding space during reduction. Default: `None` (Auto-inferred).
        n_epochs: How many dataset passes to train the projection model. Default: `None` (Auto-inferred).
        model: The model to use when generating the 2D projected embedding space layout. Possible values: `None` or `nomic-project-v1` or `nomic-project-v2`. Default: `None`.
        local_neighborhood_size: Only used when model is `nomic-project-v2`. Controls the size of the neighborhood used in the local structure optimizing step of `nomic-project-v2` algorithm. Min value: `max(n_neighbors, 1)`; max value: `128`. Default: `None` (Auto-inferred).
        spread: Determines how tight together points appear. Larger values result a more spread out point layout. Min value: `0`. It is recommended leaving this value as the default `None` (Auto-inferred).
        rho: Only used when model is nomic-project-v2. Controls the spread in the local structure optimizing step of `nomic-project-v2`. Min value: `0`; max value: `1`. It is recommended to leave this value as the default `None` (Auto-inferred).
    """

    n_neighbors: Optional[int] = None
    n_epochs: Optional[int] = None
    spread: Optional[float] = None
    local_neighborhood_size: Optional[int] = None
    model: Optional[str] = None
    rho: Optional[float] = None


class NomicTopicOptions(BaseModel):
    """
    Options for Nomic Topic Model

    Args:
        build_topic_model: If True, builds a topic model over your dataset's embeddings.
        topic_label_field: The dataset column (usually the column you embedded) that Atlas will use to assign a human-readable description to each topic.
    """

    build_topic_model: bool = True
    topic_label_field: Optional[str] = Field(default=None)
    cluster_method: str = "fast"
    enforce_topic_hierarchy: bool = False


class NomicDuplicatesOptions(BaseModel):
    """
    Options for Duplicate Detection

    Args:
        tag_duplicates: Should duplicate detection run over your datasets embeddings?
        duplicate_cutoff: A hyperparameter of duplicate detection, smaller values capture more exact duplicates.
    """

    tag_duplicates: bool = True
    duplicate_cutoff: float = DEFAULT_DUPLICATE_THRESHOLD


class NomicEmbedOptions(BaseModel):
    """
    Options for Configuring the Nomic Embedding Model

    Args:
        model: The Nomic Embedding Model to use.
    """

    model: Literal[
        "nomic-embed-text-v1",
        "nomic-embed-vision-v1",
        "nomic-embed-text-v1.5",
        "nomic-embed-vision-v1.5",
        "gte-multilingual-base",
    ] = "nomic-embed-text-v1.5"
