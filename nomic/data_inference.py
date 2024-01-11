from typing import Any, Dict, List, Optional, Union

import pyarrow as pa
from pydantic import BaseModel, Field

from .settings import (
    DEFAULT_DUPLICATE_THRESHOLD,
    DEFAULT_PROJECTION_EPOCHS,
    DEFAULT_PROJECTION_N_NEIGHBORS,
    DEFAULT_PROJECTION_SPREAD,
)


def from_list(values: Dict[str, Any], schema=None) -> pa.Table:
    tb = pa.Table.from_pylist(values, schema=schema)
    return tb


permitted_types = {
    'integer': pa.int32(),
    'float': pa.float32(),
    'date': pa.timestamp('ms'),
    'string': pa.string(),
    'categorical': pa.string(),
}


def convert_pyarrow_schema_for_atlas(schema: pa.Schema) -> pa.Schema:
    """
    Convert a pyarrow schema to one with types that match the subset of types supported by Atlas for upload.
    """
    types = {}
    whitelist = {}
    for field in schema:
        if field.name.startswith('_'):
            # Underscore fields are private to Atlas and will be handled with their own logic.
            if not field.name in {"_embeddings"}:
                raise ValueError(f"Underscore fields are reserved for Atlas internal use: {field.name}")
            whitelist[field.name] = field.type
        elif pa.types.is_boolean(field.type):
            raise TypeError(f"Boolean type not supported: {field.name}")
        elif pa.types.is_list(field.type):
            raise TypeError(f"List types not supported: {field.name}")
        elif pa.types.is_struct(field.type):
            raise TypeError(f"Struct types not supported: {field.name}")
        elif pa.types.is_dictionary(field.type):
            types[field.name] = 'categorical'
        elif pa.types.is_string(field.type):
            types[field.name] = 'string'
        elif pa.types.is_integer(field.type):
            types[field.name] = 'integer'
        elif pa.types.is_floating(field.type):
            types[field.name] = 'float'
        elif pa.types.is_timestamp(field.type):
            types[field.name] = 'date'
        elif pa.types.is_temporal(field.type):
            types[field.name] = 'date'
        else:
            raise TypeError(f"Unknown type: {field.name} {field.type}")
    usertypes = {k: permitted_types[v] for k, v in types.items()}

    return pa.schema({**usertypes, **whitelist})


class NomicProjectOptions(BaseModel):
    '''
    Options for Nomic 2D Dimensionality Reduction Model

    Args:
        n_neighbors: The number of neighbors to use when approximating the high dimensional embedding space during reduction.
        n_epochs: How many dataset passes to train the projection model.
    '''

    n_neighbors: int = DEFAULT_PROJECTION_N_NEIGHBORS
    n_epochs: int = DEFAULT_PROJECTION_EPOCHS
    spread: float = DEFAULT_PROJECTION_SPREAD


class NomicTopicOptions(BaseModel):
    '''
    Options for Nomic Topic Model

    Args:
        build_topic_model: If True, builds a topic model over your dataset's embeddings.
        topic_label_field: The dataset field/column that Atlas will use to assign a human-readable description to each topic.
    '''

    build_topic_model: bool = True
    community_description_target_field: Optional[str] = Field(None, alias='topic_label_field')
    cluster_method: str = 'fast'
    enforce_topic_hierarchy: bool = False


class NomicDuplicatesOptions(BaseModel):
    '''
    Options for Duplicate Detection

    Args:
        tag_duplicates: Should duplicate detection run over your datasets embeddings?
        duplicate_cutoff: A hyperparameter of duplicate detection, smaller values capture more exact duplicates.
    '''

    tag_duplicates: bool = True
    duplicate_cutoff: float = DEFAULT_DUPLICATE_THRESHOLD


class NomicEmbedOptions(BaseModel):
    '''
    Options for Configuring the Nomic Embedding Model

    Args:
        model: The Nomic Embedding Model to use.
    '''

    model: str = 'NomicEmbed'
