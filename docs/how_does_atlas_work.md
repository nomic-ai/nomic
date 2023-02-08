# How does Atlas work?
Atlas maps your data in three steps:

### 1. Data ingestion
Data is added to an AtlasProject
### 2. Data embedding.
When indexes the data in your project by running it through state-of-the-art neural networks to identify latent relationships.
### 3. Data organization.
Atlas projects your data into a two-dimension view, preparing it for exploration and interaction.

These steps are automatically executed when you run [`map_text`](atlas_api.md) or [`map_embeddings`](atlas_api.md) in the AtlasClient. You can see an
example of how to manually create a project, add data and create indices [here](https://github.com/nomic-ai/nomic/blob/main/examples/interactive_session.py).

Read on to learn about how projects, indices, maps interact in Atlas.

## Atlas Project
When you upload data to Atlas, your data is stored in an AtlasProject.
AtlasProject allow you to manage access control to your data and can be used to make maps.
You can make a project with the [`create_project`](atlas_api.md) method of the AtlasClient. Add data to your project with [`add_text`](atlas_api.md) or [`add_embeddings`](atlas_api.md)
methods depending on your projects data modality.

## AtlasIndex
An Atlas Index is a way of organizing data in a particular project.
To organize your data onto a map, you need to build an AtlasIndex. You can build an index on a project with the [`create_index`](atlas_api.md) method by specifying
the projects unique ID.

## AtlasMap or AtlasProjection
Maps are the interactive visualizations of your data you see in a web browser. When you build an index, you are returned a URL to
the projects map. Alternatively, find your project in the [Atlas Dashboard](https://atlas.nomic.ai/dashboard) and go
to its map.

## Organizations
Projects are owned by organizations. Organizations allow you manage granular access controls to projects.
Add your teammates to an organization to collaboratively interact with your data.

