import pandas as pd
import pyarrow as pa
from pyarrow import compute as pc
from typing import List
class AtlasDuplicates():
    """
    The duplicates 
    """
    def __init__(self, projection : "AtlasProjection"):
        self.projection = projection
        self.id_field = self.projection.project.id_field
        self._tb : pa.Table = projection._fetch_tiles().select([self.id_field, '_duplicate_class', '_cluster_id'])

    @property
    def df(self):
        """
        Returns a pandas dataframe with information about duplicate clusters and candidates.
        """
        return self.tb.to_pandas()
    
    @property
    def tb(self):
        """
        Returns a pyarrow table with information about duplicate clusters and candidates. 
        This table is memmapped from the underlying files and is the most efficient way to 
        access duplicate information.
        """
        return self._tb

    def deletion_candidates(self) -> List[str]:
        """
        Returns a list of ids for all the the duplicate candidates in the set.
        """
        dupes = self.tb[self.id_field].filter(pc.equal(self.tb['_duplicate_class'], 'deletion candidate'))
        return dupes.to_pylist()

    def __repr__(self) -> str:
        repr = f"===Atlas Duplicates for ({self.projection})===\n"
        duplicate_count = len(self.tb[self.id_field].filter(pc.equal(self.tb['_duplicate_class'], 'deletion candidate')))
        cluster_count = len(self.tb['_cluster_id'].value_counts())
        repr += f"{duplicate_count} deletion candidates in {cluster_count} clusters\n"
        return repr + self.df.__repr__()
