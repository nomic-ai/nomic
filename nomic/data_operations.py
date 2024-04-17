import base64
import concurrent
import concurrent.futures
import glob
import io
import json
import os
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas
import pandas as pd
import pyarrow as pa
import requests
from loguru import logger
from pyarrow import compute as pc
from pyarrow import feather, ipc
from tqdm import tqdm

from nomic.dataset import AtlasProjection

from .settings import EMBEDDING_PAGINATION_LIMIT
from .utils import download_feather



class AtlasMapDuplicates:
    """
    Atlas Duplicate Clusters State. Atlas can automatically group embeddings that are sufficiently close into semantic clusters.
    You can use these clusters for semantic duplicate detection allowing you to quickly deduplicate
    your data.
    """

    def __init__(self, projection: "AtlasProjection"):
        self.projection = projection
        self.id_field = self.projection.dataset.id_field
        try:
            duplicate_fields = [
                field for field in projection._fetch_tiles().column_names if "_duplicate_class" in field
            ]
            cluster_fields = [field for field in projection._fetch_tiles().column_names if "_cluster" in field]
            assert len(duplicate_fields) > 0, "Duplicate detection has not yet been run on this map."
            self.duplicate_field = duplicate_fields[0]
            self.cluster_field = cluster_fields[0]
            self._tb: pa.Table = projection._fetch_tiles().select(
                [self.id_field, self.duplicate_field, self.cluster_field]
            )
        except pa.lib.ArrowInvalid as e:
            raise ValueError("Duplicate detection has not yet been run on this map.")
        self.duplicate_field = self.duplicate_field.lstrip("_")
        self.cluster_field = self.cluster_field.lstrip("_")
        self._tb = self._tb.rename_columns([self.id_field, self.duplicate_field, self.cluster_field])

    @property
    def df(self) -> pd.DataFrame:
        """
        Pandas DataFrame mapping each data point to its cluster of semantically similar points.
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
        dupes = self.tb[self.id_field].filter(pa.compute.equal(self.tb[self.duplicate_field], 'deletion candidate'))
        return dupes.to_pylist()

    def __repr__(self) -> str:
        repr = f"===Atlas Duplicates for ({self.projection})\n"
        duplicate_count = len(
            self.tb[self.id_field].filter(pa.compute.equal(self.tb[self.duplicate_field], 'deletion candidate'))
        )
        cluster_count = len(self.tb[self.cluster_field].value_counts())
        repr += f"{duplicate_count} deletion candidates in {cluster_count} clusters\n"
        return repr + self.df.__repr__()


class AtlasMapTopics:
    """
    Atlas Topics State
    """

    def __init__(self, projection: AtlasProjection):
        self.projection = projection
        self.dataset = projection.dataset
        self.id_field = self.projection.dataset.id_field
        self._metadata = None
        self._hierarchy = None

        try:
            self._tb: pa.Table = projection._fetch_tiles()
            topic_fields = [column for column in self._tb.column_names if column.startswith("_topic_depth_")]
            self.depth = len(topic_fields)

            # If using topic ids, fetch topic labels
            if 'int' in topic_fields[0]:
                new_topic_fields = []
                metadata = self.metadata
                label_df = metadata[["topic_id", "depth", "topic_short_description"]]
                for d in range(1, self.depth + 1):
                    column = f"_topic_depth_{d}_int"
                    topic_ids_to_label = self._tb[column].to_pandas().rename('topic_id')
                    topic_ids_to_label = label_df[label_df["depth"] == d].merge(
                        topic_ids_to_label, on='topic_id', how='right'
                    )
                    new_column = f"_topic_depth_{d}"
                    self._tb = self._tb.append_column(
                        new_column, pa.Array.from_pandas(topic_ids_to_label["topic_short_description"])
                    )
                    new_topic_fields.append(new_column)
                topic_fields = new_topic_fields

            renamed_fields = [f'topic_depth_{i}' for i in range(1, self.depth + 1)]
            self._tb = self._tb.select([self.id_field] + topic_fields).rename_columns([self.id_field] + renamed_fields)

        except pa.lib.ArrowInvalid as e:
            raise ValueError("Topic modeling has not yet been run on this map.")

    @property
    def df(self) -> pandas.DataFrame:
        """
        A pandas DataFrame associating each datapoint on your map to their topics as each topic depth.
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
        Pandas DataFrame where each row gives metadata all map topics including:

        - topic id
        - a human readable topic description (topic label)
        - identifying keywords that differentiate the topic from other topics
        """
        if self._metadata is not None:
            return self._metadata

        response = requests.get(
            self.projection.dataset.atlas_api_path
            + "/v1/project/{}/index/projection/{}".format(
                self.projection.dataset.meta['id'], self.projection.projection_id
            ),
            headers=self.projection.dataset.header,
        )
        topics = json.loads(response.text)['topic_models'][0]['features']
        topic_data = [e['properties'] for e in topics]
        topic_data = pd.DataFrame(topic_data)
        column_list = [(f"_topic_depth_{i}", f"topic_depth_{i}") for i in range(1, self.depth + 1)]
        column_list.append(("topic", "topic_id"))
        topic_data = topic_data.rename(columns=dict(column_list))
        self._metadata = topic_data

        return topic_data

    @property
    def hierarchy(self) -> Dict:
        """
        A dictionary that allows iteration of the topic hierarchy. Each key is of (topic label, topic depth)
        to its direct sub-topics.
        If topic is not a key in the hierarchy, it is leaf in the topic hierarchy.
        """
        if self._hierarchy is not None:
            return self._hierarchy

        topic_df = self.metadata

        topic_hierarchy = defaultdict(list)
        cols = [f"topic_depth_{i}" for i in range(1, self.depth + 1)]

        for _, row in topic_df[cols].iterrows():
            # Only consider the non-null values for each row
            topics = [topic for topic in row if pd.notna(topic)]

            # Iterate over the topics in each row, adding each topic to the
            # list of subtopics for the topic at the previous depth
            for topic_index in range(len(topics) - 1):
                # depth is index + 1
                if topics[topic_index + 1] not in topic_hierarchy[(topics[topic_index], topic_index + 1)]:
                    topic_hierarchy[(topics[topic_index], topic_index + 1)].append(topics[topic_index + 1])
        self._hierarchy = dict(topic_hierarchy)
        return self._hierarchy

    def group_by_topic(self, topic_depth: int = 1) -> List[Dict]:
        """
        Associates topics at a given depth in the topic hierarchy to the identifiers of their contained datapoints.

        Args:
            topic_depth: Topic depth to group datums by.

        Returns:
            List of dictionaries where each dictionary contains next depth
                subtopics, subtopic ids, topic_id, topic_short_description,
                topic_long_description, and list of datum_ids.
        """

        if topic_depth > self.depth or topic_depth < 1:
            raise ValueError("Topic depth out of range.")

        # Unique datum id column to aggregate
        datum_id_col = self.dataset.meta["unique_id_field"]
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

            topic_label = topic_metadata["topic_short_description"].item()
            subtopics = []
            if (topic_label, topic_depth) in hierarchy:
                subtopics = hierarchy[(topic_label, topic_depth)]
            result_dict["subtopics"] = subtopics
            result_dict["subtopic_ids"] = topic_df[topic_df["topic_short_description"].isin(subtopics)][
                "topic_id"
            ].tolist()
            result_dict["topic_id"] = topic_metadata["topic_id"].item()
            result_dict["topic_short_description"] = topic_label
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
            A list of `{topic, count}` dictionaries, sorted from largest count to smallest count.
        '''
        data = AtlasMapData(self.projection, fields=[time_field])
        time_data = data._tb.select([self.id_field, time_field])
        merged_tb = self._tb.join(time_data, self.id_field, join_type="inner").combine_chunks()

        del time_data  # free up memory

        expr = (pc.field(time_field) >= start) & (pc.field(time_field) <= end)
        merged_tb = merged_tb.filter(expr)
        topic_densities = {}
        for depth in range(1, self.depth + 1):
            topic_column = f'topic_depth_{depth}'
            topic_counts = merged_tb.group_by(topic_column).aggregate([(self.id_field, "count")]).to_pandas()
            for _, row in topic_counts.iterrows():
                topic = row[topic_column]
                if topic not in topic_densities:
                    topic_densities[topic] = 0
                topic_densities[topic] += row[self.id_field + '_count']
        return topic_densities

    def vector_search_topics(self, queries: np.ndarray, k: int = 32, depth: int = 3) -> Dict:
        '''
        Given an embedding, returns a normalized distribution over topics.

        Useful for answering the questions such as:

        - What topic does my new datapoint belong to?
        - Does by datapoint belong to the "Dog" topic or the "Cat" topic.

        Args:
            queries: a 2d NumPy array where each row corresponds to a query vector
            k: (Default 32) the number of neighbors to use when estimating the posterior
            depth: (Default 3) the topic depth at which you want to search

        Returns:
            A dict mapping `{topic: posterior probability}` for each query.
        '''

        if queries.ndim != 2:
            raise ValueError(
                'Expected a 2 dimensional array. If you have a single query, we expect an array of shape (1, d).'
            )

        bytesio = io.BytesIO()
        np.save(bytesio, queries)

        response = requests.post(
            self.dataset.atlas_api_path + "/v1/project/data/get/embedding/topic",
            headers=self.dataset.header,
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

    Access latent (high-dimensional) and projected (two-dimensional) embeddings of your datapoints.

    ## Two-dimensional projected embeddings

    === "Accessing 2D Embeddings Example"
        ``` py
        from nomic import AtlasDataset

        project = AtlasDataset(name='My Project')
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

    ## High dimensional latent embeddings


    === "Accessing Latent Embeddings Example"
        ``` py
        from nomic import AtlasDataset

        project = AtlasDataset('My Project')
        map = project.maps[0]
        embeddings = map.embeddings.latent
        print(embeddings.shape)
        ```
    === "Output"
        ```
        [10000, 384]
        ```


    !!! warning "High dimensional embeddings"
        High dimensional embeddings are not immediately downloaded when you access the embeddings attribute - you must explicitly call `map.embeddings.latent`. Once downloaded, subsequent calls will reference your downloaded local copy.

    """

    def __init__(self, projection: "AtlasProjection"):
        self.projection = projection
        self.id_field = self.projection.dataset.id_field
        self._tb: pa.Table = projection._fetch_tiles().select([self.id_field, 'x', 'y'])
        self.dataset = projection.dataset
        self._latent = None

    @property
    def df(self):
        """
        Pandas DataFrame containing information about embeddings of your datapoints.

        Includes only the two-dimensional embeddings.
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
            Pandas DataFrame mapping your datapoints to their two-dimensional embeddings.
        """
        return self.df

    @property
    def latent(self) -> np.ndarray:
        """
        High dimensional embeddings.

        Returns:
            A memmapped NumPy array where each row contains the latent embedding of the corresponding datapoint in the same order as `map.embeddings.projected`.
        """
        if self._latent is not None:
            return self._latent

        root_embedding = self.projection.tile_destination / "0/0/0-0.embeddings.feather"
        # Not the most complete check, hence the warning below.
        if not root_embedding.exists():
            self._download_latent()
        all_embeddings = []

        for path in self.projection._tiles_in_order():
            # double with-suffix to remove '.embeddings.feather'
            files = path.parent.glob(path.with_suffix("").stem + "-*.embeddings.feather")
            # Should there be more than 10, we need to sort by int values, not string values
            sortable = sorted(files, key=lambda x: int(x.with_suffix("").stem.split("-")[-1]))
            if len(sortable) == 0:
                raise FileNotFoundError(
                    "Could not find any embeddings for tile {}".format(path)
                    + " If you possibly downloaded only some of the embeddings, run '[map_name].download_latent()'."
                )
            for file in sortable:
                tb = feather.read_table(file, memory_map=True)
                dims = tb['_embeddings'].type.list_size
                all_embeddings.append(pa.compute.list_flatten(tb['_embeddings']).to_numpy().reshape(-1, dims))
        return np.vstack(all_embeddings)

    def _download_latent(self):
        """
        Downloads the latent embeddings one file at a time.
        """
        logger.warning("Downloading latent embeddings of all datapoints.")
        limit = 10_000
        route = self.projection.dataset.atlas_api_path + '/v1/project/data/get/embedding/paged'
        last = None

        with tqdm(total=self.dataset.total_datums // limit) as pbar:
            while True:
                params = {'projection_id': self.projection.id, "last_file": last, "page_size": limit}
                r = requests.post(route, headers=self.projection.dataset.header, json=params)
                if r.status_code == 204:
                    # Download complete!
                    break
                fin = BytesIO(r.content)
                tb = feather.read_table(fin, memory_map=True)

                tilename = tb.schema.metadata[b'tile'].decode("utf-8")
                dest = (self.projection.tile_destination / tilename).with_suffix(".embeddings.feather")
                dest.parent.mkdir(parents=True, exist_ok=True)
                feather.write_feather(tb, dest)
                last = tilename
                pbar.update(1)

    def vector_search(self, queries: np.ndarray = None, ids: List[str] = None, k: int = 5) -> Dict[str, List]:
        '''
        Performs semantic vector search over data points on your map.
        If ids is specified, receive back the most similar data ids in latent vector space to your input ids.
        If queries is specified, receive back the data ids with representations most similar to the query vectors.

        You should not specify both queries and ids.

        Args:
            queries: a 2d NumPy array where each row corresponds to a query vector
            ids: a list of ids
            k: the number of closest data points (neighbors) to return for each input query/data id
        Returns:
            A tuple with two elements containing the following information:
                neighbors: A set of ids corresponding to the nearest neighbors of each query
                distances: A set of distances between each query and its neighbors.
        '''

        if queries is None and ids is None:
            raise ValueError('You must specify either a list of datum `ids` or NumPy array of `queries` but not both.')

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
            if queries.ndim != 2:
                raise ValueError(
                    'Expected a 2 dimensional array. If you have a single query, we expect an array of shape (1, d).'
                )

            bytesio = io.BytesIO()
            np.save(bytesio, queries)

            response = requests.post(
                self.projection.dataset.atlas_api_path + "/v1/project/data/get/nearest_neighbors/by_embedding",
                headers=self.projection.dataset.header,
                json={
                    'atlas_index_id': self.projection.atlas_index_id,
                    'queries': base64.b64encode(bytesio.getvalue()).decode('utf-8'),
                    'k': k,
                },
            )
        else:
            response = requests.post(
                self.projection.dataset.atlas_api_path + "/v1/project/data/get/nearest_neighbors/by_id",
                headers=self.projection.dataset.header,
                json={'atlas_index_id': self.projection.atlas_index_id, 'datum_ids': ids, 'k': k},
            )

        if response.status_code == 500:
            raise Exception('Cannot perform vector search on your map at this time. Try again later.')

        if response.status_code != 200:
            raise Exception(response.text)

        response = response.json()

        return response['neighbors'], response['distances']

    def _get_embedding_iterator(self) -> Iterable[Tuple[str, str]]:
        '''
        Deprecated in favor of `map.embeddings.latent`.

        Iterate through embeddings of your datums.

        Returns:
            An iterable mapping datum ids to their embeddings.

        '''

        raise DeprecationWarning("Deprecated as of June 2023. Iterate `map.embeddings.latent`.")

    def _download_embeddings(self, save_directory: str, num_workers: int = 10) -> bool:
        '''
        Deprecated in favor of `map.embeddings.latent`.

        Downloads embeddings to the specified save_directory.

        Args:
            save_directory: The directory to save your embeddings.
        Returns:
            True on success


        '''
        raise DeprecationWarning("Deprecated as of June 2023. Use `map.embeddings.latent`.")

    def __repr__(self) -> str:
        return str(self.df)


class AtlasMapTags:
    """
    Atlas Map Tag State. You can manipulate tags by filtering over
    the associated pandas DataFrame.
    """

    def __init__(self, projection: "AtlasProjection", auto_cleanup: Optional[bool] = False):
        self.projection = projection
        self.dataset = projection.dataset
        self.id_field = self.projection.dataset.id_field
        # Pre-fetch tiles first upon initialization
        self.projection._fetch_tiles(overwrite=False)
        self.auto_cleanup = auto_cleanup

    @property
    def df(self, overwrite: Optional[bool] = False) -> pd.DataFrame:
        '''
        Pandas DataFrame mapping each data point to its tags.
        '''
        tags = self.get_tags()
        tag_definition_ids = [tag["tag_definition_id"] for tag in tags]
        if self.auto_cleanup:
            self._remove_outdated_tag_files(tag_definition_ids)
        for tag in tags:
            self._download_tag(tag["tag_name"], overwrite=overwrite)
        tbs = []
        all_quads = list(self.projection._tiles_in_order(coords_only=True))
        for quad in tqdm(all_quads):
            quad_str = os.path.join(*[str(q) for q in quad])
            datum_id_filename = quad_str + "." + "datum_id" + ".feather"
            path = self.projection.tile_destination / Path(datum_id_filename)
            tb = feather.read_table(path, memory_map=True)
            for tag in tags:
                tag_definition_id = tag["tag_definition_id"]
                tag_filename = quad_str + "." + f"_tag.{tag_definition_id}" + ".feather"
                path = self.projection.tile_destination / Path(tag_filename)
                tag_tb = feather.read_table(path, memory_map=True)
                bitmask = None
                if "all_set" in tag_tb.column_names:
                    if tag_tb["all_set"][0].as_py() == True:
                        bitmask = pa.array([True] * len(tb), type=pa.bool_())
                    else:
                        bitmask = pa.array([False] * len(tb), type=pa.bool_())
                else:
                    bitmask = tag_tb["bitmask"]
                tb = tb.append_column(tag["tag_name"], bitmask)
            tbs.append(tb)
        return pa.concat_tables(tbs).to_pandas()

    def get_tags(self) -> Dict[str, List[str]]:
        '''
        Retrieves back all tags made in the web browser for a specific map.
        Each tag is a dictionary containing tag_name, tag_id, and metadata.

        Returns:
            A list of tags a user has created for projection.
        '''
        tags = requests.get(
            self.dataset.atlas_api_path + '/v1/project/projection/tags/get/all',
            headers=self.dataset.header,
            params={'project_id': self.dataset.id, 'projection_id': self.projection.id, 'include_dsl_rule': False},
        ).json()
        keep_tags = []
        for tag in tags:
            is_complete = requests.get(
                self.dataset.atlas_api_path + '/v1/project/projection/tags/status',
                headers=self.dataset.header,
                params={
                    'project_id': self.dataset.id,
                    'tag_id': tag["tag_id"],
                },
            ).json()['is_complete']
            if is_complete:
                keep_tags.append(tag)
        return keep_tags

    def get_datums_in_tag(self, tag_name: str, overwrite: Optional[bool] = False):
        '''
        Returns the datum ids in a given tag.

        Args:
            overwrite: If True, re-downloads the tag. Otherwise, checks to see if up
            to date tag already exists.

        Returns:
            List of datum ids.
        '''
        ordered_tag_paths = self._download_tag(tag_name, overwrite=overwrite)
        datum_ids = []
        for path in ordered_tag_paths:
            tb = feather.read_table(path)
            last_coord = path.name.split(".")[0]
            tile_path = path.with_name(last_coord + ".datum_id.feather")
            tile_tb = feather.read_table(tile_path).select([self.id_field])

            if "all_set" in tb.column_names:
                if tb["all_set"][0].as_py() == True:
                    datum_ids.extend(tile_tb[self.id_field].to_pylist())
            else:
                # filter on rows
                try:
                    tb = tb.append_column(self.id_field, tile_tb[self.id_field])
                    datum_ids.extend(tb.filter(pc.field("bitmask") == True)[self.id_field].to_pylist())
                except Exception as e:
                    raise Exception(f"Failed to fetch datums in tag. {e}")
        return datum_ids

    def _get_tag_by_name(self, name: str) -> Dict:
        """
        Returns the tag dictionary for a given tag name.
        """
        for tag in self.get_tags():
            if tag["tag_name"] == name:
                return tag
        raise ValueError(f"Tag {name} not found in projection {self.projection.id}.")

    def _download_tag(self, tag_name: str, overwrite: Optional[bool] = False):
        """
        Downloads the feather tree for large sidecar columns.
        """
        self.projection.tile_destination.mkdir(parents=True, exist_ok=True)
        root_url = f"{self.dataset.atlas_api_path}/v1/project/{self.dataset.id}/index/projection/{self.projection.id}/quadtree/"

        tag = self._get_tag_by_name(tag_name)
        tag_definition_id = tag["tag_definition_id"]

        all_quads = list(self.projection._tiles_in_order(coords_only=True))
        ordered_tag_paths = []
        for quad in tqdm(all_quads):
            quad_str = os.path.join(*[str(q) for q in quad])
            filename = quad_str + "." + f"_tag.{tag_definition_id}" + ".feather"
            path = self.projection.tile_destination / Path(filename)
            download_attempt = 0
            download_success = False
            while download_attempt < 3 and not download_success:
                download_attempt += 1
                if not path.exists() or overwrite:
                    download_feather(root_url + filename, path, headers=self.dataset.header)
                try:
                    ipc.open_file(path).schema
                    download_success = True
                except pa.ArrowInvalid:
                    path.unlink(missing_ok=True)

            if not download_success:
                raise Exception(f"Failed to download tag {tag_name}.")
            ordered_tag_paths.append(path)
        return ordered_tag_paths

    def _remove_outdated_tag_files(self, tag_definition_ids: List[str]):
        '''
        Attempts to remove outdated tag files based on tag definition ids.
        Any tag with a definition not in tag_definition_ids will be deleted.

        Args:
            tag_definition_ids: A list of tag definition ids to keep.
        '''
        # NOTE: This currently only gets triggered on `df` property
        all_quads = list(self.projection._tiles_in_order(coords_only=True))
        for quad in tqdm(all_quads):
            quad_str = os.path.join(*[str(q) for q in quad])
            tile = self.projection.tile_destination / Path(quad_str)
            tile_dir = tile.parent
            if tile_dir.exists():
                tagged_files = tile_dir.glob('*_tag*')
                for file in tagged_files:
                    tag_definition_id = file.name.split(".")[-2]
                    if tag_definition_id in tag_definition_ids:
                        try:
                            file.unlink()
                        except PermissionError:
                            print("Permission denied: unable to delete outdated tag file. Skipping")
                            return
                        except Exception as e:
                            print(f"Exception occurred when trying to delete outdated tag file: {e}. Skipping")
                            return

    def add(self, ids: List[str], tags: List[str]):
        # '''
        # Adds tags to datapoints.

        # Args:
        #     ids: The datum ids you want to tag
        #     tags: A list containing the tags you want to apply to these data points.

        # '''
        raise NotImplementedError("AtlasMapTags.add is currently not supported.")

    def remove(self, ids: List[str], tags: List[str], delete_all: bool = False) -> bool:
        # '''
        # Deletes the specified tags from the given data points.

        # Args:
        #     ids: The datum_ids to delete tags from.
        #     tags: The list of tags to delete from the data points. Each tag will be applied to all data points in `ids`.
        #     delete_all: If true, ignores ids parameter and deletes all specified tags from all data points.

        # Returns:
        #     True on success.

        # '''
        raise NotImplementedError("AtlasMapTags.remove is currently not supported.")

    def __repr__(self) -> str:
        return str(self.df)


class AtlasMapData:
    """
    Atlas Map Data (Metadata) State. This is how you can access text and other associated metadata columns
    you uploaded with your project.
    """

    def __init__(self, projection: "AtlasProjection", fields=None):
        self.projection = projection
        self.dataset = projection.dataset
        self.id_field = self.projection.dataset.id_field
        self.fields = fields
        try:
            # Run fetch_tiles first to guarantee existence of quad feather files
            self._basic_data: pa.Table = self.projection._fetch_tiles()
            sidecars = self._download_data(fields=fields)
            self._tb = self._read_prefetched_tiles_with_sidecars(sidecars)

        except pa.lib.ArrowInvalid as e:
            raise ValueError("Failed to fetch tiles for this map")

    def _read_prefetched_tiles_with_sidecars(self, additional_sidecars=None):
        tbs = []
        root = feather.read_table(self.projection.tile_destination / Path("0/0/0.feather"))
        try:
            small_sidecars = set([v for k, v in json.loads(root.schema.metadata[b"sidecars"]).items()])
        except KeyError:
            small_sidecars = set([])
        for path in self.projection._tiles_in_order():
            tb = pa.feather.read_table(path).drop(["_id", "ix", "x", "y"])
            for col in tb.column_names:
                if col[0] == "_":
                    tb = tb.drop([col])
            for sidecar_file in small_sidecars:
                carfile = pa.feather.read_table(path.parent / f"{path.stem}.{sidecar_file}.feather", memory_map=True)
                for col in carfile.column_names:
                    tb = tb.append_column(col, carfile[col])
            for big_sidecar in additional_sidecars:
                fname = base64.urlsafe_b64encode(big_sidecar.encode("utf-8")).decode("utf-8")
                carfile = pa.feather.read_table(path.parent / f"{path.stem}.{fname}.feather", memory_map=True)
                for col in carfile.column_names:
                    tb = tb.append_column(col, carfile[col])
            tbs.append(tb)
        _tb = pa.concat_tables(tbs)

        return _tb

    def _download_data(self, fields=None):
        """
        Downloads the feather tree for large sidecar columns.
        """
        self.projection.tile_destination.mkdir(parents=True, exist_ok=True)
        root = f"{self.dataset.atlas_api_path}/v1/project/{self.dataset.id}/index/projection/{self.projection.id}/quadtree/"

        all_quads = list(self.projection._tiles_in_order(coords_only=True))
        sidecars = fields
        if sidecars is None:
            sidecars = [
                field
                for field in self.dataset.dataset_fields
                if field not in self._basic_data.column_names and field != "_embeddings"
            ]
        else:
            for field in sidecars:
                assert field in self.dataset.dataset_fields, f"Field {field} not found in dataset fields."

        for quad in tqdm(all_quads):
            for sidecar in sidecars:
                quad_str = os.path.join(*[str(q) for q in quad])
                encoded_colname = base64.urlsafe_b64encode(sidecar.encode("utf-8")).decode("utf-8")
                filename = quad_str + "." + encoded_colname + ".feather"
                path = self.projection.tile_destination / Path(filename)

                if not os.path.exists(path):
                    # WARNING: Potentially large data request here
                    download_feather(root + filename, path, headers=self.dataset.header)

        return sidecars

    @property
    def df(self) -> pandas.DataFrame:
        """
        A pandas DataFrame associating each datapoint on your map to their metadata.
        Converting to pandas DataFrame may materialize a large amount of data into memory.
        """
        return self._tb.to_pandas()

    @property
    def tb(self) -> pa.Table:
        """
        Pyarrow table associating each datapoint on the map to their metadata columns.
        This table is memmapped from the underlying files and is the most efficient way to
        access metadata information.
        """
        return self._tb
