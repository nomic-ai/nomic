from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.callbacks import Callback

from nomic import AtlasUser, atlas


class AtlasLightningContainer:
    def __init__(self):
        self.embeddings = []
        self.metadata = defaultdict(list)

    def log(
        self,
        embeddings: torch.Tensor,
        metadata: Union[
            Dict[str, List],
            Dict[str, torch.Tensor],
            Dict[str, np.ndarray],
            Dict[str, int],
            Dict[str, float],
            Dict[str, str],
        ] = {},
    ):
        """Log a batch of embeddings and corresponding metadata for each embedding."""
        assert isinstance(embeddings, torch.Tensor), "You must log a torch Tensor"

        # ensure passed in embeddings are a tensor with shape (batch_size, dimension)
        if len(embeddings.shape) != 2:
            raise ValueError("Your logged embedding tensor must have shape (N,d)")

        metadata_copy = defaultdict(list)
        # sanity check the inputs
        for key in metadata:
            metadata_value = metadata[key]
            if isinstance(metadata_value, torch.Tensor):
                metadata_copy[key] = metadata_value.flatten().cpu().tolist()
            elif isinstance(metadata_value, np.ndarray):
                metadata_copy[key] = metadata_value.flatten().tolist()
            else:
                if not isinstance(metadata_value, list):
                    if (
                        isinstance(metadata_value, float)
                        or isinstance(metadata_value, int)
                        or isinstance(metadata_value, str)
                    ):
                        metadata_copy[key] = [metadata_value]

            if embeddings.shape[0] != len(metadata_copy[key]):
                raise ValueError(
                    f"Your metadata has invalid shape. You have {embeddings.shape[0]} embeddings but len(metadata['{key}']) = {len(metadata_copy[key])}"
                )

        self.embeddings.append(embeddings)
        for key in metadata:
            self.metadata[key] = self.metadata[key] + metadata_copy[key]

    def clear(self):
        """Clears all embeddings and metadata from the final produced map."""
        self.embeddings = []
        self.metadata = defaultdict(list)


class AtlasLightningModule(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.atlas: Optional[AtlasLightningContainer] = None


class AtlasEmbeddingExplorer(Callback):
    def __init__(
        self,
        max_points=-1,
        rebuild_time_delay=600,
        name=None,
        description="",
        is_public=True,
        overwrite_on_validation=False,
    ):
        """

        Args:
            max_points: The maximum points to visualize. -1 will visualize all points.
            rebuild_time_delay: Only rebuilds the embedding explorer if 'rebuild_time_delay' seconds have passed since the last rebuild.
            name: The name for your embedding explorer.
            description: A description for your embedding explorer.
            is_public: Should your embedding explorer be public
            overwrite_on_validation: Re-creates your validation set viewer on every validation run.
        """
        self.max_points = max_points
        self.name = name
        self.description = description
        self.is_public = is_public
        self.overwrite = overwrite_on_validation
        self.dataset = None
        self.map = None
        self.atlas = AtlasLightningContainer()
        self.rebuild_time_delay = rebuild_time_delay
        self.last_rebuild_timestamp = datetime.min

    def on_train_start(self, trainer: "pl.Trainer", pl_module: pl.LightningModule):
        """Verify that atlas is configured and set up variables"""
        AtlasUser()  # verify logged in.
        AtlasLightningModule(pl_module).atlas = self.atlas

    def on_sanity_check_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        AtlasUser()  # verify logged in.
        AtlasLightningModule(pl_module).atlas = AtlasLightningContainer()

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
            )
            if self.map is not None:
                logger.info(f" See your previous validation embedding space at: {self.map.map_link}")
            return
        if not self.atlas.embeddings:
            logger.info(
                "Pytorch module does not have logits recorded in _atlas_logits property, AtlasEmbeddingExplorer will not generate a map."
            )
            return
        embeddings = torch.cat(self.atlas.embeddings).detach().cpu().numpy()

        lengths = map(len, self.atlas.metadata.values())
        assert set(lengths) != 1, "Error in logging metadata, it is not all the same length"

        if self.max_points > 0:
            embeddings = embeddings[: self.max_points, :]
            for key in self.atlas.metadata:
                self.atlas.metadata[key] = self.atlas.metadata[key][: self.max_points]

                # convert all uniqish values to strings prior to insertion into Atlas so
                if len(set(self.atlas.metadata[key])) <= 10:
                    self.atlas.metadata[key] = [str(x) for x in self.atlas.metadata[key]]

        keys = list(self.atlas.metadata.keys())
        if "id" not in self.atlas.metadata:
            self.atlas.metadata["id"] = [i for i in range(embeddings.shape[0])]
            keys.append("id")

        metadata = [dict(zip(keys, vals)) for vals in zip(*(self.atlas.metadata[k] for k in keys))]

        colorable_fields = []
        for key in keys:
            if key == "id":
                continue
            if (
                isinstance(metadata[0][key], float)
                or isinstance(metadata[0][key], int)
                or key in ("class", "label", "target")
                or len(set(self.atlas.metadata[key])) <= 10
            ):
                colorable_fields.append(key)

        try:
            project = atlas.map_data(
                embeddings=embeddings,
                data=metadata,
                id_field="id",
                is_public=self.is_public,
                identifier=self.name,
                description=self.description,
                topic_model=False,
            )
        except BaseException as e:
            logger.info(e)
            logger.info("Failed to update your map on this validation epoch.")
            return
        self.dataset = project
        self.map = project.maps[0]
        self.last_rebuild_timestamp = datetime.now()
