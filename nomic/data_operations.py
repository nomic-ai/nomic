import base64
import concurrent
import concurrent.futures
import io
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas
import pandas as pd
import pyarrow as pa
import requests
from loguru import logger
from pyarrow import compute as pc
from tqdm import tqdm

from .settings import EMBEDDING_PAGINATION_LIMIT


class AtlasMapDuplicates:
    """
    Atlas Duplicate Detection State
    """

    def __init__(self, projection: "AtlasProjection"):
        self.projection = projection
        self.id_field = self.projection.project.id_field
        self._tb: pa.Table = projection._fetch_tiles().select([self.id_field, '_duplicate_class', '_cluster_id'])

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
        duplicate_count = len(
            self.tb[self.id_field].filter(pc.equal(self.tb['_duplicate_class'], 'deletion candidate'))
        )
        cluster_count = len(self.tb['_cluster_id'].value_counts())
        repr += f"{duplicate_count} deletion candidates in {cluster_count} clusters\n"
        return repr + self.df.__repr__()


class AtlasMapTopics:
    """
    Atlas Topics State
    """

    def __init__(self, projection: "AtlasProjection"):
        self.projection = projection
        self.project = projection.project
        self.id_field = self.projection.project.id_field
        self._tb: pa.Table = projection._fetch_tiles().select(
            [self.id_field, '_topic_depth_1', '_topic_depth_2', '_topic_depth_3']
        )
        self._metadata = None
        self._hierarchy = None

    @property
    def df(self) -> pandas.DataFrame:
        """
        Returns a pandas dataframe with information about topics that Atlas assigned to data points.
        """
        return self.tb.to_pandas()

    @property
    def tb(self) -> pa.Table:
        """
        Returns a pyarrow table with information associated datapoints to their Atlas assigned topics.
        This table is memmapped from the underlying files and is the most efficient way to
        access topic information.
        """
        return self._tb

    @property
    def metadata(self) -> pandas.DataFrame:
        """
        Metadata about topics.
        """
        if self._metadata is not None:
            return self._metadata

        response = requests.get(
            self.projection.project.atlas_api_path
            + "/v1/project/{}/index/projection/{}".format(
                self.projection.project.meta['id'], self.projection.projection_id
            ),
            headers=self.projection.project.header,
        )
        topics = json.loads(response.text)['topic_models'][0]['features']
        topic_data = [e['properties'] for e in topics]
        topic_data = pd.DataFrame(topic_data)
        topic_data = topic_data.rename(columns={"topic": "topic_id"})
        self._metadata = topic_data

        return topic_data

    @property
    def hierarchy(self) -> Dict:
        """
        A dictionary that allows iteration of the topic heirarchy
        """
        if self._hierarchy is not None:
            return self._hierarchy

        topic_df = self.metadata

        topic_hierarchy = defaultdict(list)
        cols = ["topic_id", "_topic_depth_1", "_topic_depth_2", "_topic_depth_3"]

        for i, row in topic_df[cols].iterrows():
            # Only consider the non-null values for each row
            topics = [topic for topic in row if pd.notna(topic)]

            # Iterate over the topics in each row, adding each topic to the
            # list of subtopics for the topic at the previous depth
            for i in range(1, len(topics) - 1):
                if topics[i + 1] not in topic_hierarchy[topics[i]]:
                    topic_hierarchy[topics[i]].append(topics[i + 1])
        self._heirarchy = dict(topic_hierarchy)

        return self._heirarchy

    def group_by_topic(self, topic_depth: int = 1) -> List[Dict]:
        """
        Group datums by topic at a set topic depth.

        Args:
            topic_depth: Topic depth to group datums by. Acceptable values
                currently are (1, 2, 3).
        Returns:
            List of dictionaries where each dictionary contains next depth
                subtopics, subtopic ids, topic_id, topic_short_description,
                topic_long_description, and list of datum_ids.
        """

        topic_cols = []
        # TODO: This will need to be changed once topic depths becomes dynamic and not hard-coded
        if topic_depth not in (1, 2, 3):
            raise ValueError("Topic depth out of range.")

        # Unique datum id column to aggregate
        datum_id_col = self.project.meta["unique_id_field"]

        df = self.df

        topic_datum_dict = df.groupby(f"_topic_depth_{topic_depth}")[datum_id_col].apply(set).to_dict()

        topic_df = self.metadata
        hierarchy = self.hierarchy

        result = []
        for topic, datum_ids in topic_datum_dict.items():
            # Encountered topic with zero datums
            if len(datum_ids) == 0:
                continue

            result_dict = {}
            topic_metadata = topic_df[topic_df["topic_short_description"] == topic]

            subtopics = hierarchy[topic]
            result_dict["subtopics"] = subtopics
            result_dict["subtopic_ids"] = topic_df[topic_df["topic_short_description"].isin(subtopics)][
                "topic_id"
            ].tolist()
            result_dict["topic_id"] = topic_metadata["topic_id"].item()
            result_dict["topic_short_description"] = topic_metadata["topic_short_description"].item()
            result_dict["topic_long_description"] = topic_metadata["topic_description"].item()
            result_dict["datum_ids"] = datum_ids
            result.append(result_dict)
        return result

    def get_topic_density(self, time_field: str, start: datetime, end: datetime):
        '''
        Counts the number of datums in each topic within a window

        Args:
            time_field: Your metadata field containing isoformat timestamps
            start: A datetime object for the window start
            end: A datetime object for the window end

        Returns:
            List[{topic: str, count: int}] - A list of {topic, count} dictionaries, sorted from largest count to smallest count
        '''
        response = requests.post(
            self.project.atlas_api_path + "/v1/project/{}/topic_density".format(self.projection.atlas_index_id),
            headers=self.project.header,
            json={'start': start.isoformat(), 'end': end.isoformat(), 'time_field': time_field},
        )
        if response.status_code != 200:
            raise Exception(response.text)

        return response.json()

    def vector_search_topics(self, queries: np.array, k: int = 32, depth: int = 3) -> Dict:
        '''
        Returns the topics best associated with each vector query

        Args:
            queries: a 2d numpy array where each row corresponds to a query vector
            k: (Default 32) the number of neighbors to use when estimating the posterior
            depth: (Default 3) the topic depth at which you want to search

        Returns:
            A dict of {topic: posterior probability} for each query
        '''

        if queries.ndim != 2:
            raise ValueError(
                'Expected a 2 dimensional array. If you have a single query, we expect an array of shape (1, d).'
            )

        bytesio = io.BytesIO()
        np.save(bytesio, queries)

        response = requests.post(
            self.project.atlas_api_path + "/v1/project/data/get/embedding/topic",
            headers=self.project.header,
            json={
                'atlas_index_id': self.projection.atlas_index_id,
                'queries': base64.b64encode(bytesio.getvalue()).decode('utf-8'),
                'k': k,
                'depth': depth,
            },
        )

        if response.status_code != 200:
            raise Exception(response.text)

        return response.json()

    def __repr__(self) -> str:
        return str(self.df)


# TODO map embeddings is incomplete.
class AtlasMapEmbeddings:
    """
    Atlas Embeddings State
    Allows you to associate datapoints with their projected (2D) and latent (ND) embeddings.
    """

    def __init__(self, projection: "AtlasProjection"):
        self.projection = projection
        self.id_field = self.projection.project.id_field
        self._tb: pa.Table = projection._fetch_tiles().select([self.id_field, 'x', 'y'])
        self.project = projection.project

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
                raise ValueError(
                    'Expected a 2 dimensional array. If you have a single query, we expect an array of shape (1, d).'
                )

            bytesio = io.BytesIO()
            np.save(bytesio, queries)

        if queries is not None:
            response = requests.post(
                self.projection.project.atlas_api_path + "/v1/project/data/get/nearest_neighbors/by_embedding",
                headers=self.projection.project.header,
                json={
                    'atlas_index_id': self.projection.atlas_index_id,
                    'queries': base64.b64encode(bytesio.getvalue()).decode('utf-8'),
                    'k': k,
                },
            )
        else:
            response = requests.post(
                self.projection.project.atlas_api_path + "/v1/project/data/get/nearest_neighbors/by_id",
                headers=self.projection.project.header,
                json={'atlas_index_id': self.projection.atlas_index_id, 'datum_ids': ids, 'k': k},
            )

        if response.status_code == 500:
            raise Exception('Cannot perform vector search on your map at this time. Try again later.')

        if response.status_code != 200:
            raise Exception(response.text)

        response = response.json()

        return response['neighbors'], response['distances']

    def get_embedding_iterator(self) -> Iterable[Tuple[str, str]]:
        '''
        Iterate through embeddings of your datums.

        Returns:
            A iterable mapping datum ids to their embeddings.

        '''

        if self.project.is_locked:
            raise Exception('Project is locked! Please wait until the project is unlocked to download embeddings')

        offset = 0
        limit = EMBEDDING_PAGINATION_LIMIT
        while True:
            response = requests.get(
                self.atlas_api_path
                + f"/v1/project/data/get/embedding/{self.project.id}/{self.projection.atlas_index_id}/{offset}/{limit}",
                headers=self.header,
            )
            if response.status_code != 200:
                raise Exception(response.text)

            content = response.json()
            if len(content['datum_ids']) == 0:
                break
            offset += len(content['datum_ids'])

            yield content['datum_ids'], content['embeddings']

    def download_embeddings(self, save_directory: str, num_workers: int = 10) -> bool:
        '''
        Downloads shards of arrow tables that map

        Args:
            save_directory: The directory to save your embeddings.
        Returns:
            True on success


        '''
        self.project._latest_project_state()

        total_datums = self.project.total_datums
        if self.project.is_locked:
            raise Exception('Project is locked! Please wait until the project is unlocked to download embeddings')

        offset = 0
        limit = EMBEDDING_PAGINATION_LIMIT

        def download_shard(offset, check_access=False):
            response = requests.get(
                self.project.atlas_api_path
                + f"/v1/project/data/get/embedding/{self.project.id}/{self.projection.atlas_index_id}/{offset}/{limit}",
                headers=self.project.header,
            )

            if response.status_code != 200:
                raise Exception(response.text)

            if check_access:
                return
            try:
                shard_name = '{}_{}_{}.feather'.format(self.projection.atlas_index_id, offset, offset + limit)
                shard_path = os.path.join(save_directory, shard_name)

                content = response.content
                is_arrow_format = content[:6] == b"ARROW1" and content[-6:] == b"ARROW1"

                if not is_arrow_format:
                    raise Exception('Expected response to be in Arrow IPC format')

                with open(shard_path, 'wb') as f:
                    f.write(content)

            except Exception as e:
                logger.error('Shard {} download failed with error: {}'.format(shard_name, e))

        download_shard(0, check_access=True)

        with tqdm(total=total_datums // limit) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(download_shard, cur_offset): cur_offset
                    for cur_offset in range(0, total_datums, limit)
                }
                for future in concurrent.futures.as_completed(futures):
                    _ = future.result()
                    pbar.update(1)

        return True

    def __repr__(self) -> str:
        raise NotImplementedError()
