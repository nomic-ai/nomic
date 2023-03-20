import torch
from pytorch_lightning.callbacks import Callback
from nomic import atlas, AtlasUser
from loguru import logger
from typing import List, Dict
from collections import defaultdict
from datetime import datetime

import numpy as np

class AtlasLightningContainer:
    def __init__(self):
        self.embeddings = []
        self.metadata = defaultdict(list)

    def log(self, embeddings: torch.Tensor, metadata: Dict[str, List] = {}):
        '''Log a batch of embeddings and corresponding metadata for each embedding.'''
        assert isinstance(embeddings, torch.Tensor), 'You must log a torch Tensor'

        #ensure passed in embeddings are a tensor with shape (batch_size, dimension)
        if len(embeddings.shape) != 2:
            raise ValueError("Your logged embedding tensor must have shape (N,d)")

        #sanity check the inputs
        for key in metadata:
            if isinstance(metadata[key], torch.Tensor):
                metadata[key] = metadata[key].flatten().cpu().tolist()
            elif isinstance(metadata[key], np.ndarray):
                metadata[key] = metadata[key].flatten().tolist()
            else:
                if not isinstance(metadata[key], list):
                    if isinstance(metadata[key], float) or isinstance(metadata[key], int) or isinstance(metadata[key], str):
                        metadata[key] = [metadata[key]]


            if embeddings.shape[0] != len(metadata[key]):
                raise ValueError(f"Your metadata has invalid shape. You have {embeddings.shape[0]} embeddings but len(metadata['{key}']) = {len(metadata[key])}")

        self.embeddings.append(embeddings)
        for key in metadata:
            self.metadata[key] = self.metadata[key] + metadata[key]

    def clear(self):
        '''Clears all embeddings and metadata from the final produced map.'''
        self.embeddings = []
        self.metadata = defaultdict(list)

class AtlasEmbeddingExplorer(Callback):
    def __init__(self, max_points=-1, rebuild_time_delay=600, name=None, description=None, is_public=True, overwrite_on_validation=False):
        '''

        Args:
            max_points: The maximum points to visualize.
            rebuild_time_delay: Only rebuilds the embedding explorer if 'rebuild_time_delay' seconds have passed since the last rebuild.
            name: The name for your embedding explorer.
            description: A description for your embedding explorer.
            is_public: Should your embedding explorer be public
            overwrite_on_validation: Re-creates your validation set viewer on every validation run.
        '''
        self.max_points = max_points
        self.name = name
        self.description = description
        self.is_public = is_public
        self.overwrite = overwrite_on_validation
        self.project = None
        self.map = None
        self.atlas = AtlasLightningContainer()
        self.rebuild_time_delay = rebuild_time_delay
        self.last_rebuild_timestamp = datetime.min

    def on_train_start(self, trainer, pl_module):
        '''Verify that atlas is configured and set up variables'''
        AtlasUser() #verify logged in.
        pl_module.atlas = self.atlas

    def on_sanity_check_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        AtlasUser()  # verify logged in.
        pl_module.atlas = AtlasLightningContainer()


    def on_train_epoch_start(self, *args, **kwargs):
        self.atlas.clear()

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.atlas.clear()

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._create_map()

    def _create_map(self):
        seconds_since_last_build = int((datetime.now() - self.last_rebuild_timestamp).total_seconds())
        if seconds_since_last_build < self.rebuild_time_delay:
            logger.info(
                f"Skipping regenerating embedding explorer for this validation epoch as its been {seconds_since_last_build} seconds since the last rebuild and `rebuild_time_delay={self.rebuild_time_delay}`"
                f" See your previous validation embedding space at: {self.map.map_link}")
            return
        if not self.atlas.embeddings:
            logger.info("Pytorch module does not have logits recorded in _atlas_logits property, AtlasEmbeddingExplorer will not generate a map.")
            return
        embeddings = torch.cat(self.atlas.embeddings).detach().cpu().numpy()

        lengths = map(len, self.atlas.metadata.values())
        assert set(lengths) != 1, 'Error in logging metadata, it is not all the same length'

        if self.max_points > 0:
            embeddings = embeddings[:self.max_points, :]
            for key in self.atlas.metadata:
                self.atlas.metadata[key] = self.atlas.metadata[key][:self.max_points]

                #convert all uniqish values to strings prior to insertion into Atlas so
                if len(set(self.atlas.metadata[key])) <= 10:
                    self.atlas.metadata[key] = [str(x) for x in self.atlas.metadata[key]]

        keys = list(self.atlas.metadata.keys())
        if 'id' not in self.atlas.metadata:
            self.atlas.metadata['id'] = [i for i in range(embeddings.shape[0])]
            keys.append('id')

        metadata = [dict(zip(keys, vals)) for vals in zip(*(self.atlas.metadata[k] for k in keys))]

        colorable_fields = []
        for key in keys:
            if key == 'id':
                continue
            if isinstance(metadata[0][key], float) or \
                    isinstance(metadata[0][key], int) or \
                    key in ('class', 'label', 'target') or \
                    len(set(self.atlas.metadata[key])) <= 10:
                colorable_fields.append(key)

        try:
            project = atlas.map_embeddings(embeddings=embeddings,
                                            data=metadata,
                                            id_field='id',
                                            colorable_fields=colorable_fields,
                                            is_public=self.is_public,
                                            name=self.name,
                                            description=self.description,
                                            reset_project_if_exists=self.overwrite,
                                            build_topic_model=False)
        except BaseException as e:
            logger.info(e)
            logger.info("Failed to update your map on this validation epoch.")
            return
        self.project = project
        self.map = project.maps[0]
        self.last_rebuild_timestamp = datetime.now()