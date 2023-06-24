import base64
import io
import pandas as pd
import concurrent
import concurrent.futures
from loguru import logger
import pyarrow as pa
from pyarrow import compute as pc
from typing import List
import requests
import numpy as np
from typing import Dict
import tqdm
import os

from .settings import EMBEDDING_PAGINATION_LIMIT

class AtlasMapDuplicates:
    """
    Atlas Duplicate Detection State
    """
    def __init__(self, projection : "AtlasProjection"):
        self.projection = projection
        self.id_field = self.projection.project.id_field
        self._tb : pa.Table = projection._fetch_tiles().select([self.id_field, '_duplicate_class', '_cluster_id'])

    @property
    def df(self) -> pd.DataFrame:
        """
        Returns a pandas dataframe with information about duplicate clusters and candidates.
        """
        return self.tb.to_pandas()
    
    @property
    def tb(self) -> pa.Table:
        """
        Returns a pyarrow table with information about duplicate clusters and candidates. 
        This table is memmapped from the underlying files and is the most efficient way to 
        access duplicate information.
        """
        return self._tb

    def deletion_candidates(self) -> List[str]:
        """
        Returns a list of ids for all the duplicate candidates in the set.

        If you remove all of these datapoints from your dataset, your dataset will be semantically deduplicated.
        """
        dupes = self.tb[self.id_field].filter(pc.equal(self.tb['_duplicate_class'], 'deletion candidate'))
        return dupes.to_pylist()

    def __repr__(self) -> str:
        repr = f"===Atlas Duplicates for ({self.projection})===\n"
        duplicate_count = len(self.tb[self.id_field].filter(pc.equal(self.tb['_duplicate_class'], 'deletion candidate')))
        cluster_count = len(self.tb['_cluster_id'].value_counts())
        repr += f"{duplicate_count} deletion candidates in {cluster_count} clusters\n"
        return repr + self.df.__repr__()


class AtlasMapTopics:
    """
    Atlas Topics State
    """

    def __init__(self, projection: "AtlasProjection"):
        self.projection = projection
        self.id_field = self.projection.project.id_field
        self._tb: pa.Table = projection._fetch_tiles().select([self.id_field, '_topic_depth_1', '_topic_depth_2', '_topic_depth_3'])

    @property
    def df(self):
        """
        Returns a pandas dataframe with information about topics that Atlas assigned to data points.
        """
        return self.tb.to_pandas()

    @property
    def tb(self):
        """
        Returns a pyarrow table with information associated datapoints to their Atlas assigned topics.
        This table is memmapped from the underlying files and is the most efficient way to
        access topic information.
        """
        return self._tb


    def __repr__(self) -> str:
        raise NotImplementedError()




#TODO map embeddings is incomplete.
class AtlasMapEmbeddings:
    """
    Atlas Embeddings State
    Allows you to associate datapoints with their projected (2D) and latent (ND) embeddings.
    """

    def __init__(self, projection: "AtlasProjection"):
        self.projection = projection
        self.id_field = self.projection.project.id_field
        self._tb: pa.Table = projection._fetch_tiles().select([self.id_field, 'x', 'y'])

    @property
    def df(self):
        """
        Returns raw representation as a pandas dataframe.
        Does not include latent embeddings.
        """
        return self.tb.to_pandas()

    @property
    def tb(self) -> pa.Table:
        """
        Returns a pyarrow table with information associated datapoints to their Atlas projected representations.
        This table is memmapped from the underlying files and is the most efficient way to
        access embedding information.

        Does not include latent embeddings
        """
        return self._tb

    @property
    def projected(self) -> pd.DataFrame:
        return self.df

    @property
    def latent(self):
        """
        #TODO
        1. download embeddings and store it in a fixed location on disk (e.g. .nomic directory)
        2. make sure the embeddings align with the arrow table download order.
        """
        raise NotImplementedError()

    def vector_search(self, queries: np.array = None, ids: List[str] = None, k: int = 5) -> Dict[str, List]:
        '''
        Performs vector similarity search over data points on your map.
        If ids is specified, receive back the most similar data ids in vector space to your input ids.
        If queries is specified, receive back the data ids with representations most similar to the query vectors.

        You should not specify both queries and ids.

        Args:
            queries: a 2d numpy array where each row corresponds to a query vector
            ids: a list of ids
            k: the number of closest data points (neighbors) to return for each input query/data id
        Returns:
            A tuple with two elements containing the following information:
                neighbors: A set of ids corresponding to the nearest neighbors of each query
                distances: A set of distances between each query and its neighbors
        '''

        if queries is None and ids is None:
            raise ValueError('You must specify either a list of datum `ids` or numpy array of `queries` but not both.')

        max_k = 128
        max_queries = 256
        if k > max_k:
            raise Exception(f"Cannot query for more than {max_k} nearest neighbors. Set `k` to {max_k} or lower")

        if ids is not None:
            if len(ids) > max_queries:
                raise Exception(f"Max ids per query is {max_queries}. You sent {len(ids)}.")
        if queries is not None:
            if not isinstance(queries, np.ndarray):
                raise Exception("`queries` must be an instance of np.array.")
            if queries.shape[0] > max_queries:
                raise Exception(f"Max vectors per query is {max_queries}. You sent {queries.shape[0]}.")

        if queries is not None:
            if queries.ndim != 2:
                raise ValueError('Expected a 2 dimensional array. If you have a single query, we expect an array of shape (1, d).')

            bytesio = io.BytesIO()
            np.save(bytesio, queries)

        if queries is not None:
            response = requests.post(
                self.projection.project.atlas_api_path + "/v1/project/data/get/nearest_neighbors/by_embedding",
                headers=self.projection.project.header,
                json={'atlas_index_id': self.projection.atlas_index_id,
                      'queries': base64.b64encode(bytesio.getvalue()).decode('utf-8'),
                      'k': k},
            )
        else:
            response = requests.post(
                self.projection.project.atlas_api_path + "/v1/project/data/get/nearest_neighbors/by_id",
                headers=self.projection.project.header,
                json={'atlas_index_id': self.projection.atlas_index_id,
                      'datum_ids': ids,
                      'k': k},
            )


        if response.status_code == 500:
            raise Exception('Cannot perform vector search on your map at this time. Try again later.')

        if response.status_code != 200:
            raise Exception(response.text)

        response = response.json()

        return response['neighbors'], response['distances']


    def __repr__(self) -> str:
        raise NotImplementedError()

