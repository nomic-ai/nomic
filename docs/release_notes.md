# Release Notes
## v1.0.25
**Tagging**: The `get_tags` method will retrieve tags you have assigned to datapoints on the map.

**Progressive projects**: You can now call the `map_*` endpoints multiple times and specify the same project each time. Doing this will add data to the project. See the [documentation](dynamic_maps.md) for examples.

**shard_size**: You can now specify a shard_size in the `map_*` endpoints. If each datum is too large, you want to use a smaller shard size to successfully send data to Atlas.

**Multilingual support**: You can now make maps that organize data language agnostically. Setting `multilingual = True` when running `map_text` will make maps agnostic to language. For example, a paragraph in English and it's translation to Spanish will be close together on the map.

## v1.0.22
**ID fields**: Every datum by default has an id field attached. You no longer have to specify an id field when mapping data.

**Bug fixes**: Numerous bugs are now squashed involving error handling.

## v1.0.14
**Progressive Maps**: Maps can now be built progressively in Atlas. See the progressive map documentation for more information.

## v1.0.13
**Documentation Improvements**: documentation was significantly altered for clarity and organization.

**Maps of text**: Atlas can now ingest your raw text data and handle the embedding for you. See the text map documentation for more detail.s
