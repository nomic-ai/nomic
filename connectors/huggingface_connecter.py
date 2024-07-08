from datasets import load_dataset
import hashlib

# Fetches data from a Hugging Face dataset
def fetch_data_from_huggingface(dataset_identifier):
    try:
        # Attempts to load dataset without specifying config
        dataset = load_dataset(dataset_identifier)
    except ValueError as e:
        # Handles error messages
        error_message = str(e)
        if "Please pick one among the available configs" in error_message:
          
            try:
                available_configs_start = error_message.index("['") + 2
                available_configs_end = error_message.index("']")
                available_configs = error_message[available_configs_start:available_configs_end].split("', '")
                # Load dataset with the first available config
                dataset = load_dataset(dataset_identifier, available_configs[0], trust_remote_code=True)
            except ValueError:
                raise ValueError("Failed to extract available configurations from the error message.")
        else:
            raise e  

    # Processes dataset entries
    data = []
    for split in dataset.keys():
        for i, example in enumerate(dataset[split]):
            # Creates a unique ID using SHA-256 for better security
            unique_str = f"{dataset_identifier}_{split}_{i}"
            short_id = hashlib.sha256(unique_str.encode()).hexdigest()[:25]
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
       
    except ValueError as e:
        print(f"Error loading dataset: {e}")



