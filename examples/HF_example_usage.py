import argparse
from huggingface_connector import hf_atlasdataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create an AtlasDataset from a Hugging Face dataset.')
    parser.add_argument('--dataset_identifier', type=str, required=True, help='The Hugging Face dataset identifier')
    parser.add_argument('--split', type=str, default="train", help='The dataset split to use (default: train)')
    parser.add_argument('--limit', type=int, default=100000, help='The maximum number of examples to load (default: 100000)')


    args = parser.parse_args()


    try:
        atlas_dataset = hf_atlasdataset(args.dataset_identifier, args.split, args.limit)
        print(f"AtlasDataset has been created for '{args.dataset_identifier}'")
    except ValueError as e:
        print(f"Error creating AtlasDataset: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")





