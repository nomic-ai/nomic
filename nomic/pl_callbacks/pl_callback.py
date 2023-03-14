import torch
from pytorch_lightning.callbacks import Callback
from nomic import atlas, AtlasUser
from loguru import logger
from typing import List, Dict
from collections import defaultdict

class AtlasLightningContainer:
    def __init__(self):
        self.embeddings = []
        self.metadata = defaultdict(list)

    def log(self, embeddings: torch.Tensor, metadata: Dict[str, List] = {}):
        '''Log a batch of embeddings and corresponding metadata for each embedding.'''
        assert isinstance(embeddings, torch.Tensor), 'You must log a torch Tensor'
        self.embeddings.append(embeddings)
        for key in metadata:
            if isinstance(metadata[key], list):
                self.metadata[key] = self.metadata[key] + metadata[key]
            else:
                self.metadata[key] = self.metadata[key] + [metadata[key]]

    def clear(self):
        '''Clears all embeddings and metadata from the final produced map.'''
        self.embeddings = []
        self.metadata = defaultdict(list)

class AtlasEmbeddingExplorer(Callback):
    def __init__(self, max_points=-1, name=None, description=None, is_public=True, overwrite_on_validation=False):
        '''

        Args:
            max_points: The maximum points to visualize.
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

    def on_train_start(self, trainer, pl_module):
        '''Verify that atlas is configured and set up variables'''
        AtlasUser() #verify logged in.
        pl_module.atlas = self.atlas

    def on_train_epoch_start(self, *args, **kwargs):
        self.atlas.clear()

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.atlas.clear()

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._create_map()

    def _create_map(self):
        if not self.atlas.embeddings:
            logger.warning("Pytorch module does not have logits recorded in _atlas_logits property, AtlasEmbeddingExplorer will not generate a map.")
            return
        embeddings = torch.cat(self.atlas.embeddings).detach().cpu().numpy()

        for key in self.atlas.metadata:
            try:
                self.atlas.metadata[key] = torch.cat(self.atlas.metadata[key]).detach().cpu().numpy()
            except BaseException:
                self.atlas.metadata[key] = self.atlas.metadata[key]

        lengths = map(len, self.atlas.metadata.values())
        assert set(lengths) != 1, 'Error in logging metadata, it is not all the same length'

        if self.max_points > 0:
            embeddings = embeddings[:self.max_points, :]
            for key in self.atlas.metadata:
                self.atlas.metadata[key] = self.atlas.metadata[key][:self.max_points]

        keys = list(self.atlas.metadata.keys())
        if 'id' not in self.atlas.metadata:
            self.atlas.metadata['id'] = [i for i in range(embeddings.shape[0])]
            keys.append('id')

        metadata = [dict(zip(keys, vals)) for vals in zip(*(self.atlas.metadata[k] for k in keys))]

        colorable_fields = []
        for key in keys:
            if key == 'id':
                continue
            if isinstance(metadata[0][key], float) or isinstance(metadata[0][key], int) or key in ('class', 'label', 'target'):
                colorable_fields.append(key)

        project = atlas.map_embeddings(embeddings=embeddings,
                                        data=metadata,
                                        id_field='id',
                                        colorable_fields=colorable_fields,
                                        is_public=self.is_public,
                                        name=self.name,
                                        description=self.description,
                                        reset_project_if_exists=self.overwrite,
                                        build_topic_model=False)
        self.project = project
        self.map = project.maps[0]