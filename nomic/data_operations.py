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
from pyarrow import ipc
from tqdm import tqdm

from .settings import EMBEDDING_PAGINATION_LIMIT


class AtlasMapDuplicates:
    """
    Atlas Duplicate Clusters State

    Atlas automatically groups embeddings that are sufficiently close into semantic clusters.
    You can use these clusters for semantic duplicate detection allowing you to quickly deduplicate
    your data.

    === "Accessing Duplicates Example"
        ``` py
        from nomic import AtlasProject

        project = AtlasProject(name='My Project')
        map = project.maps[0]
        print(map.duplicates)
        ```
    === "Output"
        ```
        460 deletion candidates in 9540 clusters
              id_      duplicate_class   cluster_id
        0      0A            singleton         5178
        1      0g  retention candidate          271
        2      0Q            singleton         6672
        3      0w            singleton         7529
        4      1A            singleton         1587
        ...   ...                  ...          ...
        9999  JZU            singleton         6346
        ```
    """

    def __init__(self, projection: "AtlasProjection"):
        self.projection = projection
        self.id_field = self.projection.project.id_field
        try:
            self._tb: pa.Table = projection._fetch_tiles().select([self.id_field, '_duplicate_class', '_cluster_id'])
        except pa.lib.ArrowInvalid as e:
            raise ValueError("Duplicate detection has not yet been run on this map.")
        self._tb = self._tb.rename_columns([self.id_field, 'duplicate_class', 'cluster_id'])

    @property
    def df(self) -> pd.DataFrame:
        """
        Pandas dataframe mapping each data point to its cluster of semantically similar points

        === "Accessing Duplicates Example"
            ``` py
            from nomic import AtlasProject

            project = AtlasProject(name='My Project')
            map = project.maps[0]
            print(map.duplicates.df)
            ```
        === "Output"
            ```
                  id_     _duplicate_class  _cluster_id
            0      0A            singleton         5178
            1      0g  retention candidate          271
            2      0Q            singleton         6672
            3      0w            singleton         7529
            4      1A            singleton         1587
            ...   ...                  ...          ...
            9999  JZU            singleton         6346
            ```
        """
        return self.tb.to_pandas()

    @property
    def tb(self) -> pa.Table:
        """
        Pyarrow table with information about duplicate clusters and candidates.
        This table is memmapped from the underlying files and is the most efficient way to
        access duplicate information.
        """
        return self._tb

    def deletion_candidates(self) -> List[str]:
        """

        Returns:
            The ids for all data points which are semantic duplicates and are candidates for being deleted from the dataset. If you remove these data points from your dataset, your dataset will be semantically deduplicated.
        """
        dupes = self.tb[self.id_field].filter(pc.equal(self.tb['duplicate_class'], 'deletion candidate'))
        return dupes.to_pylist()

    def __repr__(self) -> str:
        repr = f"===Atlas Duplicates for ({self.projection})===\n"
        duplicate_count = len(
            self.tb[self.id_field].filter(pc.equal(self.tb['duplicate_class'], 'deletion candidate'))
        )
        cluster_count = len(self.tb['cluster_id'].value_counts())
        repr += f"{duplicate_count} deletion candidates in {cluster_count} clusters\n"
        return repr + self.df.__repr__()


class AtlasMapTopics:
    """
    Atlas Topics State

    === "Accessing Topics Example"
        ``` py
        from nomic import AtlasProject

        project = AtlasProject(name='My Project')
        map = project.maps[0]
        print(map.topics)
        ```
    === "Output"
        ```
                        id_      topic_depth_1       topic_depth_2          topic_depth_3
        0     000262a5-2811  Space exploration      Hurricane Jeanne        Spacecraft Cassini
        1     000c453d-ee97   English football      Athens 2004 Olympics    bobby rathore
        ...
        9999  fffcc65c-38dc  Space exploration      Presidential elections  Blood
        ```
    """

    def __init__(self, projection: "AtlasProjection"):
        self.projection = projection
        self.project = projection.project
        self.id_field = self.projection.project.id_field
        try:
            self._tb: pa.Table = projection._fetch_tiles().select(
                [self.id_field, '_topic_depth_1', '_topic_depth_2', '_topic_depth_3']
            ).rename_columns([self.id_field, 'topic_depth_1', 'topic_depth_2', 'topic_depth_3'])
        except pa.lib.ArrowInvalid as e:
            raise ValueError("Topic modeling has not yet been run on this map.")
        self._metadata = None
        self._hierarchy = None

    @property
    def df(self) -> pandas.DataFrame:
        """
        A pandas dataframe associating each datapoint on your map to their topics as each topic depth.
        """
        return self.tb.to_pandas()

    @property
    def tb(self) -> pa.Table:
        """
        Pyarrow table associating each datapoint on the map to their Atlas assigned topics.
        This table is memmapped from the underlying files and is the most efficient way to
        access topic information.
        """
        return self._tb

    @property
    def metadata(self) -> pandas.DataFrame:
        """
        Pandas dataframe where each row gives metadata all map topics including:

        - topic id
        - a human readable topic description
        - identifying keywords that differentiate the topic from other topics
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
        topic_data = topic_data.rename(columns={"topic": "topic_id",
                                                '_topic_depth_1': 'topic_depth_1',
                                                '_topic_depth_2': 'topic_depth_2',
                                                '_topic_depth_3': 'topic_depth_3'})
        self._metadata = topic_data

        return topic_data

    @property
    def hierarchy(self) -> Dict:
        """
        A dictionary that allows iteration of the topic hierarchy. Each key is a topic mapping to its sub-topics.
        If topic is not a key in the hierarchy, it is leaf in the topic hierarchy.
        """
        if self._hierarchy is not None:
            return self._hierarchy

        topic_df = self.metadata

        topic_hierarchy = defaultdict(list)
        cols = ["topic_id", "topic_depth_1", "topic_depth_2", "topic_depth_3"]

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
        Associates topics at a given depth in the topic hierarchy to the identifiers of their contained datapoints.

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

        topic_datum_dict = df.groupby(f"topic_depth_{topic_depth}")[datum_id_col].apply(set).to_dict()

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
        Computes the density/frequency of topics in a given interval of a timestamp field.

        Useful for answering questions such as:

        - What topics increased in prevalence between December and January?

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
        Given an embedding, returns a normalized distribution over topics.

        Useful for answering the questions such as:

        - What topic does my new datapoint belong to?
        - Does by datapoint belong to the "Dog" topic or the "Cat" topic.

        Args:
            queries: a 2d numpy array where each row corresponds to a query vector
            k: (Default 32) the number of neighbors to use when estimating the posterior
            depth: (Default 3) the topic depth at which you want to search

        Returns:
            A dict mapping {topic: posterior probability} for each query
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

class AtlasMapEmbeddings:
    """
    Atlas Embeddings State

    Access latent (high-dimensional) and projected (two-dimensional) embeddings of your datapoints. High dimensional embeddings
    are not immediately downloaded when you access the embeddings attribute.

    === "Accessing Embeddings Example"
        ``` py
        from nomic import AtlasProject

        project = AtlasProject(name='My Project')
        map = project.maps[0]
        print(map.embeddings)
        ```
    === "Output"
        ```
              id_          x          y
        0      0A  -6.164423  21.517719
        1      0g  -6.606402  -5.601104
        2      0Q  -9.206946   7.448542
        ...   ...        ...        ...
        9998  JZQ   2.110881 -12.937058
        9999  JZU   7.865006  -6.876243
        ```
    """

    def __init__(self, projection: "AtlasProjection"):
        self.projection = projection
        self.id_field = self.projection.project.id_field
        self._tb: pa.Table = projection._fetch_tiles().select([self.id_field, 'x', 'y'])
        self.project = projection.project

    @property
    def df(self):
        """
        Pandas dataframe containing information about embeddings of your datapoints.

        Includes only the two-dimensional embeddings
        """
        return self.tb.to_pandas()

    @property
    def tb(self) -> pa.Table:
        """
        Pyarrow table containing two-dimensional embeddings of each of your data points.
        This table is memmapped from the underlying files and is the most efficient way to
        access embedding information.

        Does not include high-dimensional embeddings.
        """
        return self._tb

    @property
    def projected(self) -> pd.DataFrame:
        """
        Two-dimensional embeddings.

        These are the points you see in your web browser.

        Returns:
            Pandas dataframe mapping your datapoints to their two-dimensional embeddings.
        """
        return self.df

    @property
    def latent(self):
        # """
        # #TODO
        # 1. download embeddings and store it in a fixed location on disk (e.g. .nomic directory)
        # 2. make sure the embeddings align with the arrow table download order.
        # """
        raise NotImplementedError(
            "Accessing latent embeddings is not yet implemented. You must use map.download_embeddings() method for now."
        )

    def vector_search(self, queries: np.array = None, ids: List[str] = None, k: int = 5) -> Dict[str, List]:
        '''
        Performs semantic vector search over data points on your map.
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
        Downloads embeddings to the specified save_directory.

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

            shard_name = '{}_{}_{}.feather'.format(self.projection.atlas_index_id, offset, offset + limit)
            shard_path = os.path.join(save_directory, shard_name)
            try:
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
        return str(self.df)


class AtlasMapTags:
    """
    Atlas Map Tag State

    Tags are shared across all maps in your AtlasProject. You can manipulate tags by filtering over
    the associated pandas dataframe

    === "Accessing Tags Example"
        ``` py
        from nomic import AtlasProject

        project = AtlasProject(name='My Project')
        map = project.maps[0]
        print(map.tags)
        ```
    === "Output"
        ```
              id_  oil  search_engines
        0      0A    0               0
        1      0g    0               0
        2      0Q    0               0
        3      0w    0               0
        4      1A    1               0
        ...   ...  ...             ...
        9998  JZQ    0               0
        9999  JZU    0               0
        ```
    """

    def __init__(self, projection: "AtlasProjection"):
        self.projection = projection
        self.project = projection.project
        self.id_field = self.projection.project.id_field
        self._tb: pa.Table = projection._fetch_tiles().select([self.id_field])

    @property
    def df(self) -> pd.DataFrame:
        """
        Pandas dataframe mapping each data point to its tags.

        === "Accessing Tags Example"
            ``` py
            from nomic import AtlasProject

            project = AtlasProject(name='My Project')
            map = project.maps[0]
            print(map.tags.df)
            ```
        === "Output"
            ```
                  id_  oil  search_engines
            0      0A    0               0
            1      0g    0               0
            2      0Q    0               0
            3      0w    0               0
            4      1A    1               0
            ...   ...  ...             ...
            9998  JZQ    0               0
            9999  JZU    0               0
            ```
        """

        id_frame = self._tb.to_pandas()
        tag_to_datums = self.get_tags()

        # encoded contains a multi-hot vector withs 1 for all rows that contain that tag
        encoded = {key: [] for key in list(tag_to_datums.keys())}
        for id in id_frame[self.id_field]:
            for key in encoded:
                if id in tag_to_datums[key]:
                    encoded[key].append(1)
                else:
                    encoded[key].append(0)

        tag_frame = pandas.DataFrame(encoded)

        return pd.concat([id_frame, tag_frame], axis=1)

    def get_tags(self) -> Dict[str, List[str]]:
        '''
        Retrieves back all tags made in the web browser for a specific map

        Returns:
            A dictionary mapping data points to tags.
        '''
        # now get the tags
        datums_and_tags = requests.post(
            self.project.atlas_api_path + '/v1/project/tag/read/all_by_datum',
            headers=self.project.header,
            json={
                'project_id': self.project.id,
            },
        ).json()['results']

        label_to_datums = {}
        for item in datums_and_tags:
            for label in item['labels']:
                if label not in label_to_datums:
                    label_to_datums[label] = set()
                label_to_datums[label].add(item['datum_id'])
        return label_to_datums

    def add(self, ids: List[str], tags: List[str]):
        '''
        Adds tags to datapoints.

        Args:
            ids: The datum ids you want to tag
            tags: A list containing the tags you want to apply to these data points.

        '''
        assert isinstance(ids, list), 'ids must be a list of strings'
        assert isinstance(tags, list), 'tags must be a list of strings'

        colname = json.dumps(
            {
                'project_id': self.project.id,
                'atlas_index_id': self.projection.atlas_index_id,
                'type': 'datum_id',
                'tags': tags,
            }
        )
        payload_table = pa.table([pa.array(ids, type=pa.string())], [colname])
        buffer = io.BytesIO()
        writer = ipc.new_file(buffer, payload_table.schema, options=ipc.IpcWriteOptions(compression='zstd'))
        writer.write_table(payload_table)
        writer.close()
        payload = buffer.getvalue()

        headers = self.project.header.copy()
        headers['Content-Type'] = 'application/octet-stream'
        response = requests.post(self.project.atlas_api_path + "/v1/project/tag/add", headers=headers, data=payload)
        if response.status_code != 200:
            raise Exception("Failed to add tags")

    def remove(self, ids: List[str], tags: List[str], delete_all: bool = False) -> bool:
        '''
        Deletes the specified tags from the given data points.

        Args:
            ids: The datum_ids to delete tags from.
            tags: The list of tags to delete from the data points. Each tag will be applied to all data points in `ids`.
            delete_all: If true, ignores ids parameter and deletes all specified tags from all data points.

        Returns:
            True on success

        '''
        assert isinstance(ids, list), 'datum_ids must be a list of strings'
        assert isinstance(tags, list), 'tags must be a list of strings'

        colname = json.dumps(
            {
                'project_id': self.project.id,
                'atlas_index_id': self.projection.atlas_index_id,
                'type': 'datum_id',
                'tags': tags,
                'delete_all': delete_all,
            }
        )
        payload_table = pa.table([pa.array(ids, type=pa.string())], [colname])
        buffer = io.BytesIO()
        writer = ipc.new_file(buffer, payload_table.schema, options=ipc.IpcWriteOptions(compression='zstd'))
        writer.write_table(payload_table)
        writer.close()
        payload = buffer.getvalue()

        headers = self.project.header.copy()
        headers['Content-Type'] = 'application/octet-stream'
        response = requests.post(self.project.atlas_api_path + "/v1/project/tag/delete", headers=headers, data=payload)
        if response.status_code != 200:
            raise Exception("Failed to delete tags")

    def __repr__(self) -> str:
        return str(self.df)
