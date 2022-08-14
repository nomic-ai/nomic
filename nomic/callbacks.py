import torch
from pytorch_lightning.callbacks import Callback
from .atlas import AtlasClient
from loguru import logger

class NomicMappingCallback(Callback):
    def __init__(self, max_points=10000, map_name=None, map_description=None):
        self.max_points = max_points
        self.map_name = map_name
        self.map_description = map_description

    def on_train_start(self, trainer, pl_module):
        atlas = AtlasClient()
        atlas._get_current_user()

    def on_train_end(self, trainer, pl_module):
        if not hasattr(pl_module, 'nomic_logits') or not hasattr(pl_module, 'nomic_labels'):
            logger.warning("Pytorch module does not have logits recorded, Nomic will not generate a map.")
            return

        atlas = AtlasClient()
        embeddings = torch.cat(pl_module.nomic_logits).detach().cpu().numpy()
        labels = torch.cat(pl_module.nomic_labels).detach().cpu().numpy()

        if self.max_points > 0:
            embeddings = embeddings[:self.max_points, :]
            labels = labels[:self.max_points]

        metadata = []
        for idx, label in enumerate(labels):
            metadata.append({'label': str(label), 'id': idx})

        response = atlas.map_embeddings(embeddings=embeddings,
                                        data=metadata,
                                        id_field='id',
                                        colorable_fields=['label'],
                                        is_public=True,
                                        map_name=self.map_name,
                                        map_description=self.map_description,
                                        organization_name=None,  # defaults to your current user.
                                        num_workers=20)