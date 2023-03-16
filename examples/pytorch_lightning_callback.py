import os
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from nomic.pl_callbacks import AtlasEmbeddingExplorer
import nomic

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
torch.manual_seed(0)
# #api key to a limited demo account. Make your own account at atlas.nomic.ai
nomic.login('7xDPkYXSYDc1_ErdTPIcoAR9RNd8YDlkS3nVNXcVoIMZ6')



class MNISTModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)
        self.l2 = torch.nn.Linear(10, 10)

    def forward(self, x):
        return torch.relu(self.l2(torch.relu(self.l1(x.view(x.size(0), -1)))))

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        prediction = torch.argmax(logits, dim=1)

        #an image for each embedding
        image_links = [f'https://s3.amazonaws.com/static.nomic.ai/mnist/eval/{label}/{batch_idx*BATCH_SIZE+idx}.jpg'
                       for idx, label in enumerate(y)]
        metadata = {'label': y, 'prediction': prediction, 'url': image_links}
        self.atlas.log(embeddings=logits, metadata=metadata)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


mnist_model = MNISTModel()

# Init DataLoader from MNIST Dataset
train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# Initialize a trainer
max_epochs = 20

# Initialize the Embedding Explorer 🗺️ hook
embedding_explorer = AtlasEmbeddingExplorer(max_points=10_000,
                                            name="MNIST Validation Latent Space",
                                            description="MNIST Validation Latent Space",
                                            overwrite_on_validation=True)
trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,
    max_epochs=max_epochs,
    check_val_every_n_epoch=10,
    callbacks=[TQDMProgressBar(refresh_rate=20),
               embedding_explorer],
)

# Train the model ⚡
trainer.fit(mnist_model, train_dataloaders=train_loader, val_dataloaders=test_loader)


trainer.validate(mnist_model, test_loader)


print(embedding_explorer.map)
