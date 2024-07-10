from datasets import load_dataset, get_dataset_split_names
from nomic import AtlasDataset
import numpy as np
import pandas as pd
import pyarrow as pa


# Gets data from HF dataset
def get_hfdata(dataset_identifier):
    try:
        # Loads dataset without specifying config
        dataset = load_dataset(dataset_identifier)
    except ValueError as e:
        # Grabs available configs and loads dataset using it
        configs = get_dataset_split_names(dataset_identifier)
        config = configs[0]
        dataset = load_dataset(dataset_identifier, config, trust_remote_code=True, streaming=True, split=config + "[:100000]")


    # Processes dataset entries using Arrow
    id_counter = 0
    data = []
    for split in dataset.keys():
        for example in dataset[split]:
            # Adds a sequential ID
            example['id'] = str(id_counter)
            id_counter += 1
            data.append(example)


    # Convert the data list to an Arrow table
    table = pa.Table.from_pandas(pd.DataFrame(data))


    return table


# Converts booleans, lists etc to strings
def convert_to_string(value):
    if isinstance(value, bool):
        return str(value)
    elif isinstance(value, list):
        return ' '.join(map(convert_to_string, value))  
    elif isinstance(value, np.ndarray):
        return ' '.join(map(str, value.flatten()))
    elif hasattr(value, 'tolist'):
        return ' '.join(map(str, value.tolist()))
    else:
        return str(value)


# Processes Arrow table and converts necessary fields to strings
def process_table(table):
    # Converts columns with complex types to strings
    for col in table.schema.names:
        column = table[col].to_pandas()
        if column.dtype == np.bool_ or column.dtype == object or isinstance(column[0], (list, np.ndarray)):
            column = column.apply(convert_to_string)
            table = table.set_column(table.schema.get_field_index(col), col, pa.array(column))


    return table


# Creates AtlasDataset from HF dataset
def hf_atlasdataset(dataset_identifier):
    table = get_hfdata(dataset_identifier.strip())


    map_name = dataset_identifier.replace('/', '_')
    if not table:
        raise ValueError("No data was found for the provided dataset.")


    dataset = AtlasDataset(
        map_name,
        unique_id_field="id",
    )


    # Ensures all values are converted to strings
    processed_table = process_table(table)


    # Adds data to the AtlasDataset
    dataset.add_data(data=processed_table.to_pandas().to_dict(orient='records'))


    return dataset





