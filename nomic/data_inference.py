import pyarrow as pa
from typing import List, Union, Dict, Any

"""

Atlas stores and transfers data using a subset of the Apache Arrow standard.
Pyarrow is used to convert python, pandas, and numpy data types to Arrow types;
you can also pass any Arrow table (created by polars, duckdb, pyarrow, etc.) directly to Atlas
and the types will be automatically converted.

Before being uploaded, all data is converted with the following rules:
* Strings are converted to Arrow strings and stored as UTF-8.
* Integers are converted to 32-bit integers. (In the case that you have larger integers, they are probably either IDs, in which case you should convert them to strings;
or they are a field that you want perform analysis on, in which case you should convert them to floats.)
* Floats are converted to 32-bit (single-precision) floats.
* Embeddings, regardless of precision, are uploaded as 16-bit (half-precision) floats, and stored in Arrow as FixedSizeList.
* All dates and datetimes are converted to Arrow timestamps with millisecond precision and no time zone.
  (If you have a use case that requires timezone information or micro/nanosecond precision, please let us know.)
* Categorical types (called 'dictionary' in Arrow) are supported, but values stored as categorical must be strings.

Other data types (including booleans, binary, lists, and structs) are not supported.
Values stored as a dictionary must be strings.

All fields besides embeddings and the user-specified ID field are nullable.

"""

def from_list(values: Dict[str, Any], schema = None) -> pa.Table:
  tb = pa.Table.from_pylist(values, schema=schema)
  return tb

permitted_types = {
    'integer': pa.int32(),
    'float': pa.float32(),
    'date': pa.timestamp('ms'),
    'string': pa.string(),
    'categorical': pa.string(),
}

def convert_pyarrow_schema_for_atlas(schema : pa.Schema) -> pa.Schema:
  """
  Convert a pyarrow schema to one with types that match the subset of types supported by Atlas for upload.
  """
  types = {}
  whitelist = {}
  for field in schema:      
      if field.name.startswith('_'):
          # Underscore fields are private to Atlas and will be handled with their own logic.
          if not field.name in {"_embeddings"}:
            raise ValueError(f"Underscore fields are reserved for Atlas internal use: {field.name}")
          whitelist[field.name] = field.type
      elif pa.types.is_boolean(field.type):
          raise TypeError(f"Boolean type not supported: {field.name}")
      elif pa.types.is_list(field.type):
          raise TypeError(f"List types not supported: {field.name}")
      elif pa.types.is_struct(field.type):
          raise TypeError(f"Struct types not supported: {field.name}")
      elif pa.types.is_dictionary(field.type):
          types[field.name] = 'categorical'
      elif pa.types.is_string(field.type):
          types[field.name] = 'string'
      elif pa.types.is_integer(field.type):
          types[field.name] = 'integer'
      elif pa.types.is_floating(field.type):
          types[field.name] = 'float'
      elif pa.types.is_timestamp(field.type):
          types[field.name] = 'date'
      elif pa.types.is_temporal(field.type):
          types[field.name] = 'date'
      else:
          raise TypeError(f"Unknown type: {field.name} {field.type}")
  usertypes = {
    k: permitted_types[v] for k, v in types.items()
  }

  return pa.schema({**usertypes, **whitelist})
