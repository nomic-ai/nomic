from datasets import load_dataset, get_dataset_split_names
from nomic import AtlasDataset
import pyarrow as pa
import pyarrow.compute as pc

# Gets data from HF dataset
def get_hfdata(dataset_identifier, split="train", limit=100000):
    splits = get_dataset_split_names(dataset_identifier)
    dataset = load_dataset(dataset_identifier, split=split, streaming=True)

    if not dataset:
        raise ValueError("No dataset was found for the provided identifier and split.")

    # Processes dataset entries using Arrow
    id_counter = 0
    data = []
    for example in dataset:
        # Adds a sequential ID
        example['id'] = str(id_counter)
        id_counter += 1
        data.append(example)

    # Convert the data list to an Arrow table
    table = pa.Table.from_pylist(data)
    return table

# Function to convert complex types to strings using Arrow
def process_table(table):
    # Converts columns with complex types to strings
    for col in table.schema.names:
        column = table[col]
        if pa.types.is_boolean(column.type):
            table = table.set_column(table.schema.get_field_index(col), col, pc.cast(column, pa.string()))
        elif pa.types.is_list(column.type):
            new_column = []
            for item in column:
                if pa.types.is_struct(column.type.value_type):
                    # Flatten the struct and cast as string for each row
                    flattened = ", ".join(str(sub_item.as_py()) for sub_item in item.values)
                    new_column.append(flattened)
                else:
                    new_column.append(str(item))
            table = table.set_column(table.schema.get_field_index(col), col, pa.array(new_column, pa.string()))
        elif pa.types.is_dictionary(column.type):
            table = table.set_column(table.schema.get_field_index(col), col, pc.cast(column, pa.string()))
    return table

# Creates AtlasDataset from HF dataset
def hf_atlasdataset(dataset_identifier, split="train", limit=100000):
    table = get_hfdata(dataset_identifier.strip(), split, limit)
    map_name = dataset_identifier.replace('/', '_')
    if table.num_rows == 0:
        raise ValueError("No data was found for the provided dataset.")

    dataset = AtlasDataset(
        map_name,
        unique_id_field="id",
    )

    # Process the table to ensure all complex types are converted to strings
    processed_table = process_table(table)

    # Add data to the AtlasDataset
    dataset.add_data(data=processed_table)

    return dataset


