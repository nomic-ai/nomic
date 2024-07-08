from datasets import load_dataset
import hashlib

# Gets data from a Hugging Face dataset with automatic configuration
def fetch_data_from_huggingface(dataset_identifier):
    try:
        # Try loading the dataset without specifying a configuration
        dataset = load_dataset(dataset_identifier, trust_remote_code=True)
    except ValueError as e:
        # If there is an error it might be because of the config selection
        if "Please pick one among the available configs" in str(e):
            # Gets available config and selects first one
            available_configs = str(e).split("['")[1].split("']")[0].split("', '")
            dataset = load_dataset(dataset_identifier, available_configs[0], trust_remote_code=True)
        else:
            raise e

    data = []
    for split in dataset.keys():
        for i, example in enumerate(dataset[split]):
            # Creates a unique and shortened ID
            unique_str = f"{dataset_identifier}_{split}_{i}"
            short_id = hashlib.sha1(unique_str.encode()).hexdigest()[:25]
            example['id'] = short_id
            data.append(example)
    return data

# Main load function to be used as a connector
def load(dataset_identifier):
    data = fetch_data_from_huggingface(dataset_identifier.strip())

    if data:
        return data
    else:
        raise ValueError("No data was found for the provided dataset.")
if __name__ == "__main__":
    dataset_identifier = input("Enter Hugging Face dataset identifier: ").strip()

    try:
        data = load(dataset_identifier)
        print(f"Dataset has been loaded successfully. Number of entries: {len(data)}")
        # You can add more processing logic here if needed
    except ValueError as e:
        print(f"Error loading dataset: {e}")


