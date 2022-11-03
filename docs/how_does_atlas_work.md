# How does Atlas work?
For centuries, geographic atlases have provided collections of maps essential to navigating complex physical landscapes.

Atlas brings that same spirit to the complexity of your data. You can build & explore maps of your data in three steps:

### 1. Data ingestion
Atlas ingests your data into a Project.
### 2. Data embedding.
When you build an Index over a project, Atlas will automatically run your projects data through advanced neural networks
which identify how datums in your project relate to one another.
### 3. Data organization.
Atlas then organizes the embeddings of your projects datums. This organization allows you to quickly search and
view your data on a two-dimensional map.

These steps are automatically executed when you run [`map_text`](atlas_client.md) or [`map_embeddings`](atlas_client.md) in the AtlasClient. You can see an
example of how to manually create a project, add data and create indices [here](https://github.com/nomic-ai/nomic/blob/main/examples/interactive_session.py).

Read on to learn about how projects, indices, maps interact in Atlas.

## Project
Think of a Project as the container in which all related work is stored: datasets, maps generated from the data, & their respective access controls.
You can make a project with the [`create_project`](atlas_client.md) method of the AtlasClient. Add data to your project with [`add_text`](atlas_client.md) or [`add_embeddings`](atlas_client.md)
methods depending on your projects data modality.

## Indices
Indices are ways of organizing the data in a particular project.
To organize your data onto a map, you need to build an index. You can build an index on a project with the [`create_index`](atlas_client.md) method by specifying
the projects unique ID.

You likely recognize the term "index" from contexts like the creating indeces on a database table. Here, we use the term specifically to mean [what?].

## Maps
Just a geographic maps allow you to navigate the physical features of a landscape, Maps in Nomic form the primary aftifact by which you navigate the landscape of your data. 
Pracitcally speaking, Maps are artifacts produced when your build an index on a project. When you build an index, you are returned a URL to
the projects map. Alternatively, find your project on your [Atlas Dashboard](https://atlas.nomic.ai/dashboard) and go
to its map.

## Organizations
Projects are owned by organizations. Organizations allow you manage granular access controls to projects.
Add your teammates to an organization to collaboratively interact with and explore your data.

