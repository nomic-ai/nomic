from datasets import load_dataset
from nomic import AtlasDataset
from ulid import ULID
import numpy as np

# Gets data from HF dataset
def get_hfdata(dataset_identifier):
    try:
        # Loads dataset without specifying config
        dataset = load_dataset(dataset_identifier)
    except ValueError as e:
        # Handles error messages
        error_message = str(e)
        if "Please pick one among the available configs" in error_message:
            try:
                available_configs_start = error_message.index("['") + 2
                available_configs_end = error_message.index("']")
                available_configs = error_message[available_configs_start:available_configs_end].split("', '")
                dataset = load_dataset(dataset_identifier, available_configs[0], trust_remote_code=True)
            except ValueError:
                raise ValueError("Failed to get available configurations")
        else:
            raise e


    # Processes dataset entries
    data = []
    for split in dataset.keys():
        for i, example in enumerate(dataset[split]):
            # Creates a unique ULID
            ulid = ULID()
            example['id'] = str(ulid)
            data.append(example)


    return data

# Creates AtlasDataset from HF dataset
def hf_atlasdataset(dataset_identifier):
    data = get_hfdata(dataset_identifier.strip())


    map_name = dataset_identifier.replace('/', '_')
    if not data:
        raise ValueError("No data was found for the provided dataset.")


    dataset = AtlasDataset(
        map_name,
        unique_id_field="id",
    )


    # Convert all booleans and lists to strings
    for entry in data:
        for key, value in entry.items():
            if isinstance(value, bool):
                entry[key] = str(value)
            elif isinstance(value, list):
                entry[key] = ' '.join(map(str, value))
            elif isinstance(value, np.ndarray):
                entry[key] = ' '.join(map(str, value.flatten()))
            elif hasattr(value, 'tolist'):
                entry[key] = ' '.join(map(str, value.tolist()))
            else:
                entry[key] = str(value)


    dataset.add_data(data=data)


    return dataset





