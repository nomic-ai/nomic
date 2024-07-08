from datasets import load_dataset
from ulid import ULID  


#need to add an init.py

# Function to fetch data from a Hugging Face dataset
def fetch_data_from_huggingface(dataset_identifier, dataset_split=None):
    try:
        # Attempt to load the dataset without specifying a configuration
        dataset = load_dataset(dataset_identifier)
        
        if dataset_split is None:
            # Use the first available split by default
            split = next(iter(dataset.keys()))
        else:
            # Use the specified split
            split = dataset_split
        
        data = [] 
      
        for i, example in enumerate(dataset[split]):
            # Create a unique ULID
            ulid = ULID()
            example['id'] = str(ulid)
            data.append(example)
            
        return data

    except ValueError as e:
        # Handle error messages
        error_message = str(e)
        if "Please pick one among the available configs" in error_message:
            try:
                available_configs_start = error_message.index("['") + 2
                available_configs_end = error_message.index("']")
                available_configs = error_message[available_configs_start:available_configs_end].split("', '")
                # Load dataset with the first available config
                dataset = load_dataset(dataset_identifier, available_configs[0], trust_remote_code=True)
                # Proceed with data loading as above
                split = next(iter(dataset.keys()))
                data = []
                for example in enumerate(dataset[split]):
                    ulid = ULID()
                    example['id'] = str(ulid)
                    data.append(example)
                    
                return data
            
            except ValueError:
                raise ValueError("Failed to extract available configurations from the error message.")
        
        else:
            raise e  # Re-raise other ValueErrors

# Main load function to be used as a connector
def load(dataset_identifier, dataset_split=None):
    data = fetch_data_from_huggingface(dataset_identifier.strip(), dataset_split)

    if data:
        return data
    else:
        raise ValueError("No data was found for the provided dataset.")




