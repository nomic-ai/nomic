from datasets import load_dataset
from nomic import AtlasDataset
from tqdm import tqdm

dataset = load_dataset('ChihHsuan-Yang/Arboretum', split='train[:100000]')
ids = list(range(len(dataset)))
dataset = dataset.add_column("id", ids)

atlas_dataset = AtlasDataset("andriy/arboretum-100k-image-url-upload", unique_id_field="id")
records = dataset.remove_columns(["photo_id"]).to_list()

records = [record for record in tqdm(records) if record["photo_url"] is not None]
image_urls = [record.pop("photo_url") for record in records]

atlas_dataset.add_data(data=records, blobs=image_urls)
atlas_dataset.create_index(embedding_model="nomic-embed-vision-v1.5", topic_model=False)