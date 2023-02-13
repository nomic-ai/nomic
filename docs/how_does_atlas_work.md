# How does Atlas work?
Atlas maps your data in three steps:

### 1. Data ingestion
Data is added to an AtlasProject.
### 2. Data embedding.
Atlas vectorizes the data in your project by running it through state-of-the-art neural networks to identify latent relationships.
### 3. Data organization.
Atlas organizes your data on a 2D plane, preparing it for exploration and interaction.

These steps are automatically execute when you run [`map_text`](atlas_api.md) or [`map_embeddings`](atlas_api.md).

## Atlas Concepts

## AtlasProject
When you upload data to Atlas, your data is stored in an AtlasProject.
An AtlasProject allows you to manage access control to your data and organize it into maps.
You can make a project by creating an instance of the [AtlasProject](atlas_api.md) class. Add data to your project with [`add_text`](atlas_api.md) or [`add_embeddings`](atlas_api.md)
methods depending on your projects data modality.


## AtlasProjection
An AtlasProjection corresponds to a map built with the data in an AtlasProject. Any state you modify on a map in your
web browser can be accessed programmatically through an AtlasProjection.
When you build a map, you can find it in the `projections` property of your AtlasProject.
Alternatively, find your project in the [Atlas Dashboard](https://atlas.nomic.ai/dashboard).

## Organizations
Projects are owned by organizations. Organizations allow you manage granular access controls to projects.
Add your teammates to an organization to collaboratively interact with your dataset.

