"""
pip install "pytorch-lightning>=1.4" "torch" "torchvision" "torchmetrics>=0.6"
"""

import os

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
from nomic.callbacks import NomicMappingCallback

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64

class MNISTModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)
        self.nomic_logits = []
        self.nomic_labels = []

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        if self.trainer.max_epochs-1 == self.trainer.current_epoch: #on last epoch
            self.nomic_logits.append(logits)
            self.nomic_labels.append(y)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


mnist_model = MNISTModel()

# Init DataLoader from MNIST Dataset
train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

# Initialize a trainer
max_epochs = 3
trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=max_epochs,
    callbacks=[TQDMProgressBar(refresh_rate=20),
               NomicMappingCallback(max_points=10000, map_name="MNIST Map", map_description=f"Sample of MNIST MLP logits after {max_epochs} epochs of training.")],
)

# Train the model ⚡
trainer.fit(mnist_model, train_loader)