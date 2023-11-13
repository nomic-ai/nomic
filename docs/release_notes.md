# Release Notes

## v2.0.14

Improvements for stability and scale.

## v2.0.0

Atlas map state is now accessible as top level attributes of the AtlasProjection ("the Map") class.

```python
from atlas import AtlasProject

project = AtlasProject(name='My Project')
map = project.maps[0]

```

- Topics (`map.topics`)
- Embeddings (`map.embeddings`)
- Semantic Duplicate Clusters (`map.duplicates`)
- Tagging (`map.tags`)

The `Accessing Atlas State` section of the documentation shows some of the ways you can use this data.

## v1.1.7

Raise correct errors on bad id uploads

## v1.1.0

### New Data validation

1. Uploads are now internally handled as Arrow tables, allowing greater type safety and data throughput.
2. In addition to passing lists of dicts, you can directly pass pandas dataframes or pyarrow tables to any upload methods.
3. Datetime formats are now passed as native python dates or datetimes (or as pandas date or datetime). ISO-formatted strings will no longer be automatically coerced--just pass your own.
4. Null values are now allowed in any fields except for embeddings and ids. These can be passed either by setting the key to None, omiting a key from a dictionary, or using a pandas null type.
5. Typechecking is stricter than before, with the aim of raising errors on the client side sooner.

### Deprecations

- `shard_size` and `num_workers` are deprecated.

## v1.0.25

**Tagging**: The `get_tags` method will retrieve tags you have assigned to datapoints on the map.

**Progressive projects**: You can now call the `map_*` endpoints multiple times and specify the same project each time. Doing this will add data to the project. See the [documentation](dynamic_maps.md) for examples.

**shard_size**: You can now specify a shard*size in the `map*\*` endpoints. If each datum is too large, you want to use a smaller shard size to successfully send data to Atlas.

## v1.0.22

**ID fields**: Every datum by default has an id field attached. You no longer have to specify an id field when mapping data.

**Bug fixes**: Numerous bugs are now squashed involving error handling.

## v1.0.14

**Progressive Maps**: Maps can now be built progressively in Atlas. See the progressive map documentation for more information.

## v1.0.13

**Documentation Improvements**: documentation was significantly altered for clarity and organization.

**Maps of text**: Atlas can now ingest your raw text data and handle the embedding for you. See the text map documentation for more detail.s
