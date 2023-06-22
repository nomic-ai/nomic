import pandas as pd
import concurrent
import concurrent.futures
from loguru import logger
import pyarrow as pa
from pyarrow import compute as pc
from typing import List
import requests
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


    def _download_embeddings(self, save_directory: str, num_workers: int = 10) -> bool:
        '''
        Downloads shards of arrow tables that map

        Args:
            save_directory: The directory to save your embeddings.
        Returns:
            True on success


        '''
        self.projection.project._latest_project_state()

        total_datums = self.project.total_datums
        if self.projection.project.is_locked:
            raise Exception('Project is locked! Please wait until the project is unlocked to download embeddings')

        offset = 0
        limit = EMBEDDING_PAGINATION_LIMIT

        def download_shard(offset, check_access=False):
            response = requests.get(
                self.project.atlas_api_path + f"/v1/project/data/get/embedding/{self.projection.project.id}/{self.projection.atlas_index_id}/{offset}/{limit}",
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

