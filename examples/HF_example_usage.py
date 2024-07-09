
from huggingface_connecter import hf_atlasdataset
import logging

if __name__ == "__main__":
    dataset_identifier = input("Enter Hugging Face dataset identifier: ").strip()


    try:
        atlas_dataset = hf_atlasdataset(dataset_identifier)
        logging.info(f"AtlasDataset has been created for '{dataset_identifier}'")
    except ValueError as e:
        logging.error(f"Error creating AtlasDataset: {e}")











