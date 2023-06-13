import pandas as pd

class AtlasDuplicates(pd.DataFrame):
    def __init__(self, projection : AtlasProjection):
        self.projection = projection
        projection._fetch_tiles().select(['_duplicate_class', '_duplicate_cluster_id'])