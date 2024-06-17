import base64
import io
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import requests
from loguru import logger
from pyarrow import compute as pc
from pyarrow import feather
from tqdm import tqdm


class AtlasMapDuplicates:
    """
    Atlas Duplicate Clusters State. Atlas can automatically group embeddings that are sufficiently close into semantic clusters.
    You can use these clusters for semantic duplicate detection allowing you to quickly deduplicate
    your data.
    """

    def __init__(self, projection: "AtlasProjection"):  # type: ignore
        self.projection = projection
        self.id_field = self.projection.dataset.id_field

        duplicate_columns = [
            (field, sidecar)
            for field, sidecar in self.projection._registered_columns
            if field.startswith("_duplicate_class")
        ]
        cluster_columns = [
            (field, sidecar) for field, sidecar in self.projection._registered_columns if field.startswith("_cluster")
        ]

        assert len(duplicate_columns) > 0, "Duplicate detection has not yet been run on this map."

        self._duplicate_column = duplicate_columns[0]
        self._cluster_column = cluster_columns[0]
        self._tb = None

    def _load_duplicates(self):
        """
        Loads duplicates from the feather tree.
        """
        tbs = []
        duplicate_sidecar = self._duplicate_column[1]
        self.duplicate_field = self._duplicate_column[0].lstrip("_")
        self.cluster_field = self._cluster_column[0].lstrip("_")
        logger.info("Loading duplicates")
        for key in tqdm(self.projection._manifest["key"].to_pylist()):
            # Use datum id as root table
            tb = feather.read_table(
                self.projection.tile_destination / Path(key).with_suffix(".datum_id.feather"), memory_map=True
            )
            path = self.projection.tile_destination

            if duplicate_sidecar == "":
                path = path / Path(key).with_suffix(".feather")
            else:
                path = path / Path(key).with_suffix(f".{duplicate_sidecar}.feather")

            duplicate_tb = feather.read_table(path, memory_map=True)
            for field in (self._duplicate_column[0], self._cluster_column[0]):
                tb = tb.append_column(field, duplicate_tb[field])
            tbs.append(tb)
        self._tb = pa.concat_tables(tbs).rename_columns([self.id_field, self.duplicate_field, self.cluster_field])

    def _download_duplicates(self):
        """
        Downloads the feather tree for duplicates.
        """
        logger.info("Downloading duplicates")
        self.projection._download_sidecar("datum_id", overwrite=False)
        assert self._cluster_column[1] == self._duplicate_column[1], "Cluster and duplicate should be in same sidecar"
        self.projection._download_sidecar(self._duplicate_column[1], overwrite=False)

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
        if isinstance(self._tb, pa.Table):
            return self._tb
        self._download_duplicates()
        self._load_duplicates()
        return self._tb

    def deletion_candidates(self) -> List[str]:
        """

        Returns:
            The ids for all data points which are semantic duplicates and are candidates for being deleted from the dataset. If you remove these data points from your dataset, your dataset will be semantically deduplicated.
        """
        dupes = self.tb[self.id_field].filter(pa.compute.equal(self.tb[self.duplicate_field], "deletion candidate"))  # type: ignore
        return dupes.to_pylist()

    def __repr__(self) -> str:
        repr = f"===Atlas Duplicates for ({self.projection})\n"
        duplicate_count = len(
            self.tb[self.id_field].filter(pa.compute.equal(self.tb[self.duplicate_field], "deletion candidate"))  # type: ignore
        )
        cluster_count = len(self.tb[self.cluster_field].value_counts())
        repr += f"{duplicate_count} deletion candidates in {cluster_count} clusters\n"
        return repr + self.df.__repr__()


class AtlasMapTopics:
    """
    Atlas Topics State
    """

    def __init__(self, projection: "AtlasProjection"):  # type: ignore
        self.projection = projection
        self.dataset = projection.dataset
        self.id_field = self.projection.dataset.id_field
        self._metadata = None
        self._hierarchy = None
        self._topic_columns = [
            column for column in self.projection._registered_columns if column[0].startswith("_topic_depth_")
        ]
        assert len(self._topic_columns) > 0, "Topic modeling has not yet been run on this map."
        self.depth = len(self._topic_columns)
        self._tb = None

    def _load_topics(self):
        """
        Loads topics from the feather tree.
        """
        integer_topics = False
        # pd.Series to match pd typing
        label_df: Optional[Union[pd.DataFrame, pd.Series]] = None
        if "int" in self._topic_columns[0][0]:
            integer_topics = True
            label_df = self.metadata[["topic_id", "depth", "topic_short_description"]]
        tbs = []
        # Should just be one sidecar
        topic_sidecar = set([sidecar for _, sidecar in self._topic_columns]).pop()
        logger.info("Loading topics")
        for key in tqdm(self.projection._manifest["key"].to_pylist()):
            # Use datum id as root table
            tb = feather.read_table(
                self.projection.tile_destination / Path(key).with_suffix(".datum_id.feather"), memory_map=True
            )
            path = self.projection.tile_destination
            if topic_sidecar == "":
                path = path / Path(key).with_suffix(".feather")
            else:
                path = path / Path(key).with_suffix(f".{topic_sidecar}.feather")

            topic_tb = feather.read_table(path, memory_map=True)
            # Do this in depth order
            for d in range(1, self.depth + 1):
                column = f"_topic_depth_{d}"
                if integer_topics:
                    column = f"_topic_depth_{d}_int"
                    topic_ids_to_label = topic_tb[column].to_pandas().rename("topic_id")
                    assert label_df is not None
                    topic_ids_to_label = pd.DataFrame(label_df[label_df["depth"] == d]).merge(
                        topic_ids_to_label, on="topic_id", how="right"
                    )
                    new_column = f"_topic_depth_{d}"
                    tb = tb.append_column(
                        new_column, pa.Array.from_pandas(topic_ids_to_label["topic_short_description"])
                    )
                else:
                    tb = tb.append_column(f"_topic_depth_1", topic_tb["_topic_depth_1"])
            tbs.append(tb)

        renamed_columns = [self.id_field] + [f"topic_depth_{i}" for i in range(1, self.depth + 1)]
        self._tb = pa.concat_tables(tbs).rename_columns(renamed_columns)

    def _download_topics(self):
        """
        Downloads the feather tree for topics.
        """
        logger.info("Downloading topics")
        self.projection._download_sidecar("datum_id", overwrite=False)
        topic_sidecars = set([sidecar for _, sidecar in self._topic_columns])
        assert len(topic_sidecars) == 1, "Multiple topic sidecars found."
        self.projection._download_sidecar(topic_sidecars.pop(), overwrite=False)

    @property
    def df(self) -> pd.DataFrame:
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
        if isinstance(self._tb, pa.Table):
            return self._tb
        self._download_topics()
        self._load_topics()
        return self._tb

    @property
    def metadata(self) -> pd.DataFrame:
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
                self.projection.dataset.meta["id"], self.projection.projection_id
            ),
            headers=self.projection.dataset.header,
        )
        topics = json.loads(response.text)["topic_models"][0]["features"]
        topic_data = [e["properties"] for e in topics]
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
        """
        Computes the density/frequency of topics in a given interval of a timestamp field.

        Useful for answering questions such as:

        - What topics increased in prevalence between December and January?

        Args:
            time_field: Your metadata field containing isoformat timestamps
            start: A datetime object for the window start
            end: A datetime object for the window end

        Returns:
            A list of `{topic, count}` dictionaries, sorted from largest count to smallest count.
        """
        data = AtlasMapData(self.projection, fields=[time_field])
        time_data = data.tb.select([self.id_field, time_field])
        merged_tb = self.tb.join(time_data, self.id_field, join_type="inner").combine_chunks()

        del time_data  # free up memory

        expr = (pc.field(time_field) >= start) & (pc.field(time_field) <= end)
        merged_tb = merged_tb.filter(expr)
        topic_densities = {}
        for depth in range(1, self.depth + 1):
            topic_column = f"topic_depth_{depth}"
            topic_counts = merged_tb.group_by(topic_column).aggregate([(self.id_field, "count")]).to_pandas()
            for _, row in topic_counts.iterrows():
                topic = row[topic_column]
                if topic not in topic_densities:
                    topic_densities[topic] = 0
                topic_densities[topic] += row[self.id_field + "_count"]
        return topic_densities

    def vector_search_topics(self, queries: np.ndarray, k: int = 32, depth: int = 3) -> Dict:
        """
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
        """

        if queries.ndim != 2:
            raise ValueError(
                "Expected a 2 dimensional array. If you have a single query, we expect an array of shape (1, d)."
            )

        bytesio = io.BytesIO()
        np.save(bytesio, queries)

        response = requests.post(
            self.dataset.atlas_api_path + "/v1/project/data/get/embedding/topic",
            headers=self.dataset.header,
            json={
                "atlas_index_id": self.projection.atlas_index_id,
                "queries": base64.b64encode(bytesio.getvalue()).decode("utf-8"),
                "k": k,
                "depth": depth,
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

    def __init__(self, projection: "AtlasProjection"):  # type: ignore
        self.projection = projection
        self.id_field = self.projection.dataset.id_field
        self.dataset = projection.dataset
        self._tb: pa.Table = None
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
        if isinstance(self._tb, pa.Table):
            return self._tb

        self._download_projected()

        logger.info("Loading projected embeddings")

        tbs = []
        coord_sidecar = self.projection._get_sidecar_from_field("x")
        for key in tqdm(self.projection._manifest["key"].to_pylist()):
            # Use datum id as root table
            tb = feather.read_table(
                self.projection.tile_destination / Path(key).with_suffix(".datum_id.feather"), memory_map=True
            )
            path = self.projection.tile_destination
            if coord_sidecar == "":
                path = path / Path(key).with_suffix(".feather")
            else:
                path = path / Path(key).with_suffix(f".{coord_sidecar}.feather")
            carfile = feather.read_table(path, memory_map=True)
            for col in carfile.column_names:
                if col in ["x", "y"]:
                    tb = tb.append_column(col, carfile[col])
            tbs.append(tb)
        self._tb = pa.concat_tables(tbs)
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

        downloaded_files_in_tile_order = self._download_latent()
        assert len(downloaded_files_in_tile_order) > 0, "No embeddings found for this map."
        all_embeddings = []
        logger.info("Loading latent embeddings")
        for path in tqdm(downloaded_files_in_tile_order):
            tb = feather.read_table(path, memory_map=True)
            dims = tb["_embeddings"].type.list_size
            all_embeddings.append(pa.compute.list_flatten(tb["_embeddings"]).to_numpy().reshape(-1, dims))  # type: ignore
        return np.vstack(all_embeddings)

    def _download_projected(self) -> List[Path]:
        """
        Downloads the feather tree for projection coordinates.
        """
        logger.info("Downloading projected embeddings")
        # Note that y coord should be in same sidecar
        coord_sidecar = self.projection._get_sidecar_from_field("x")
        self.projection._download_sidecar("datum_id", overwrite=False)
        return self.projection._download_sidecar(coord_sidecar, overwrite=False)

    def _download_latent(self) -> List[Path]:
        """
        Downloads the feather tree for embeddings.
        Returns the path to downloaded embeddings.
        """
        # TODO: Is size of the embedding files (several hundreds of MBs) going to be a problem here?
        logger.info("Downloading latent embeddings")
        embedding_sidecar = None
        for field, sidecar in self.projection._registered_columns:
            # NOTE: be _embeddings or _embedding
            if field == "_embeddings":
                embedding_sidecar = sidecar
                break

        if embedding_sidecar is None:
            raise ValueError("No embeddings found for this map.")
        return self.projection._download_sidecar(embedding_sidecar, overwrite=False)

    def vector_search(
        self, queries: Optional[np.ndarray] = None, ids: Optional[List[str]] = None, k: int = 5
    ) -> Tuple[List, List]:
        """
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
        """

        if queries is None and ids is None:
            raise ValueError("You must specify either a list of datum `ids` or NumPy array of `queries` but not both.")

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
                    "Expected a 2 dimensional array. If you have a single query, we expect an array of shape (1, d)."
                )

            bytesio = io.BytesIO()
            np.save(bytesio, queries)

            response = requests.post(
                self.projection.dataset.atlas_api_path + "/v1/project/data/get/nearest_neighbors/by_embedding",
                headers=self.projection.dataset.header,
                json={
                    "atlas_index_id": self.projection.atlas_index_id,
                    "queries": base64.b64encode(bytesio.getvalue()).decode("utf-8"),
                    "k": k,
                },
            )
        else:
            response = requests.post(
                self.projection.dataset.atlas_api_path + "/v1/project/data/get/nearest_neighbors/by_id",
                headers=self.projection.dataset.header,
                json={"atlas_index_id": self.projection.atlas_index_id, "datum_ids": ids, "k": k},
            )

        if response.status_code == 500:
            raise Exception("Cannot perform vector search on your map at this time. Try again later.")

        if response.status_code != 200:
            raise Exception(response.text)

        response = response.json()

        return response["neighbors"], response["distances"]

    def _get_embedding_iterator(self) -> Iterable[Tuple[str, str]]:
        """
        Deprecated in favor of `map.embeddings.latent`.

        Iterate through embeddings of your datums.

        Returns:
            An iterable mapping datum ids to their embeddings.

        """

        raise DeprecationWarning("Deprecated as of June 2023. Iterate `map.embeddings.latent`.")

    def _download_embeddings(self, save_directory: str, num_workers: int = 10) -> bool:
        """
        Deprecated in favor of `map.embeddings.latent`.

        Downloads embeddings to the specified save_directory.

        Args:
            save_directory: The directory to save your embeddings.
        Returns:
            True on success


        """
        raise DeprecationWarning("Deprecated as of June 2023. Use `map.embeddings.latent`.")

    def __repr__(self) -> str:
        return str(self.df)


class AtlasMapTags:
    """
    Atlas Map Tag State. You can manipulate tags by filtering over
    the associated pandas DataFrame.
    """

    def __init__(self, projection: "AtlasProjection", auto_cleanup: Optional[bool] = False):  # type: ignore
        self.projection = projection
        self.dataset = projection.dataset
        self.id_field = self.projection.dataset.id_field
        # Pre-fetch datum ids first upon initialization
        try:
            self.projection._download_sidecar("datum_id")
        except Exception:
            raise ValueError("Failed to fetch datum ids which is required to load tags.")
        self.auto_cleanup = auto_cleanup

    @property
    def df(self, overwrite: bool = False) -> pd.DataFrame:
        """
        Pandas DataFrame mapping each data point to its tags.
        """
        tags = self.get_tags()
        tag_definition_ids = [tag["tag_definition_id"] for tag in tags]
        if self.auto_cleanup:
            self._remove_outdated_tag_files(tag_definition_ids)
        for tag in tags:
            self._download_tag(tag["tag_name"], overwrite=overwrite)
        tbs = []
        logger.info("Loading tags")
        for key in tqdm(self.projection._manifest["key"].to_pylist()):
            datum_id_path = self.projection.tile_destination / Path(key).with_suffix(".datum_id.feather")
            tb = feather.read_table(datum_id_path, memory_map=True)
            for tag in tags:
                tag_definition_id = tag["tag_definition_id"]
                path = self.projection.tile_destination / Path(key).with_suffix(f"._tag.{tag_definition_id}.feather")
                tag_tb = feather.read_table(path, memory_map=True)
                bitmask = None
                if "all_set" in tag_tb.column_names:
                    bool_v = tag_tb["all_set"][0].as_py() == True
                    bitmask = pa.array([bool_v] * len(tb), type=pa.bool_())
                else:
                    bitmask = tag_tb["bitmask"]
                tb = tb.append_column(tag["tag_name"], bitmask)
            tbs.append(tb)
        return pa.concat_tables(tbs).to_pandas()

    def get_tags(self) -> List[Dict[str, str]]:
        """
        Retrieves back all tags made in the web browser for a specific map.
        Each tag is a dictionary containing tag_name, tag_id, and metadata.

        Returns:
            A list of tags a user has created for projection.
        """
        tags = requests.get(
            self.dataset.atlas_api_path + "/v1/project/projection/tags/get/all",
            headers=self.dataset.header,
            params={"project_id": self.dataset.id, "projection_id": self.projection.id, "include_dsl_rule": False},
        ).json()
        keep_tags = []
        for tag in tags:
            is_complete = requests.get(
                self.dataset.atlas_api_path + "/v1/project/projection/tags/status",
                headers=self.dataset.header,
                params={
                    "project_id": self.dataset.id,
                    "tag_id": tag["tag_id"],
                },
            ).json()["is_complete"]
            if is_complete:
                keep_tags.append(tag)
        return keep_tags

    def get_datums_in_tag(self, tag_name: str, overwrite: bool = False):
        """
        Returns the datum ids in a given tag.

        Args:
            overwrite: If True, re-downloads the tag. Otherwise, checks to see if up
            to date tag already exists.

        Returns:
            List of datum ids.
        """
        tag_paths = self._download_tag(tag_name, overwrite=overwrite)
        datum_ids = []
        for path in tag_paths:
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

    def _download_tag(self, tag_name: str, overwrite: bool = False):
        """
        Downloads the feather tree for large sidecar columns.
        """
        logger.info("Downloading tags")
        tag = self._get_tag_by_name(tag_name)
        tag_definition_id = tag["tag_definition_id"]
        return self.projection._download_sidecar(f"_tag.{tag_definition_id}", overwrite=overwrite)

    def _remove_outdated_tag_files(self, tag_definition_ids: List[str]):
        """
        Attempts to remove outdated tag files based on tag definition ids.
        Any tag with a definition not in tag_definition_ids will be deleted.

        Args:
            tag_definition_ids: A list of tag definition ids to keep.
        """
        # NOTE: This currently only gets triggered on `df` property
        for key in self.projection._manifest["key"].to_pylist():
            tile = self.projection.tile_destination / Path(key)
            tile_dir = tile.parent
            if tile_dir.exists():
                tag_files = tile_dir.glob("*_tag*")
                for file in tag_files:
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

    def __init__(self, projection: "AtlasProjection", fields=None):  # type: ignore
        self.projection = projection
        self.dataset = projection.dataset
        self.id_field = self.projection.dataset.id_field
        if fields is None:
            # TODO: fall back on something more reliable here
            self.fields = self.dataset.dataset_fields
        else:
            for field in fields:
                assert field in self.dataset.dataset_fields, f"Field {field} not found in dataset fields."
            self.fields = fields
        self._tb = None

    def _load_data(self, data_columns: List[Tuple[str, str]]):
        """
        Loads data from a list of data columns (field and sidecar name tuples).

        Args:
            data_columns: A list of tuples containing field name and sidecar name.
        """
        tbs = []

        sidecars_to_load = set([sidecar for _, sidecar in data_columns if sidecar != "datum_id"])
        logger.info("Loading data")
        for key in tqdm(self.projection._manifest["key"].to_pylist()):
            # Use datum id as root table
            tb = feather.read_table(
                self.projection.tile_destination / Path(key).with_suffix(".datum_id.feather"), memory_map=True
            )
            for sidecar in sidecars_to_load:
                path = self.projection.tile_destination
                if sidecar == "":
                    path = path / Path(key).with_suffix(".feather")
                else:
                    path = path / Path(key).with_suffix(f".{sidecar}.feather")
                carfile = feather.read_table(path, memory_map=True)
                for col in carfile.column_names:
                    if col in self.fields:
                        tb = tb.append_column(col, carfile[col])
            tbs.append(tb)

        self._tb = pa.concat_tables(tbs)

    def _download_data(self, fields: Optional[List[str]] = None) -> List[Tuple[str, str]]:
        """
        Downloads the feather tree for user uploaded data.

        fields:
            A list of fields to download. If None, downloads all fields.

        Returns:
            List of downloaded columns
        """
        logger.info("Downloading data")
        self.projection.tile_destination.mkdir(parents=True, exist_ok=True)

        # Download specified or all sidecar fields + always download datum_id
        data_columns_to_load = [
            (str(field), str(sidecar))
            for field, sidecar in self.projection._registered_columns
            if field[0] != "_" and ((field in fields) or sidecar == "datum_id")
        ]

        # TODO: less confusing progress bar
        for sidecar in set([sidecar for _, sidecar in data_columns_to_load]):
            self.projection._download_sidecar(sidecar)
        return data_columns_to_load

    @property
    def df(self) -> pd.DataFrame:
        """
        A pandas DataFrame associating each datapoint on your map to their metadata.
        Converting to pandas DataFrame may materialize a large amount of data into memory.
        """
        logger.warning("Converting to pandas dataframe. This may materialize a large amount of data into memory.")
        return self.tb.to_pandas()

    @property
    def tb(self) -> pa.Table:
        """
        Pyarrow table associating each datapoint on the map to their metadata columns.
        This table is memmapped from the underlying files and is the most efficient way to
        access metadata information.
        """
        if isinstance(self._tb, pa.Table):
            return self._tb

        columns = self._download_data(fields=self.fields)
        self._load_data(columns)
        return self._tb
