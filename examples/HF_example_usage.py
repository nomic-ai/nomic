
from nomic_connector import hf_atlasdataset


if __name__ == "__main__":
    dataset_identifier = input("Enter Hugging Face dataset identifier: ").strip()


    try:
        atlas_dataset = hf_atlasdataset(dataset_identifier)
        print(f"AtlasDataset has been created for '{dataset_identifier}'")
    except ValueError as e:
        print(f"Error creating AtlasDataset: {e}")









