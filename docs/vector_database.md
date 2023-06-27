# Visualizing a Vector Database

Atlas is an interactive visual layer and debugger for vector databases.

This tutorial will show you how you can visualize your Weaviate and Pinecone vector databases with Atlas.

## Why Visualize A Vector Database

Vector databases allow you to query your data semantically by indexing embedding vectors. By interactively visualizing embeddings, you can quickly understand the space of possible query results from your vector database and find bad embeddings which may produce poor query results.

## Weaviate

!!! warning "Required Properties"
    When adding data to your weaviate database be sure to include the additional properties of id and vectors this can be done by adding this code when importing data to the database: `_additional = {"vector", "id"}`

First you need your Atlas API Token and a Weaviate Database URL.
If your database requires more authorization add it to the client object.

```python
import weaviate
from nomic import AtlasProject
import numpy as np
import nomic

nomic.login("NOMIC API KEY")

client = weaviate.Client(
    url="WEAVIATE DATABASE URL",
)
```

Next we'll gather all of the classes and their respective properties from the database.
To do this we will iterate through the database schema and append the classes and properties list.

```python
schema = client.schema.get()

classes = []
props = []
for c in schema["classes"]:
    classes.append(c["class"])
    temp = []
    for p in c["properties"]:
        if p["dataType"] == ["text"]:
            temp.append(p["name"])
    props.append(temp)
```

Now we will make a helper function, this will allow us to map classes that are larger than 10,000 data points. 
It queries the database while allowing us to use a cursor to store our place.
```python
def get_batch_with_cursor(
    client, class_name, class_properties, batch_size, cursor=None
):
    query = (
        client.query.get(class_name, class_properties)
        .with_additional(["vector", "id"])
        .with_limit(batch_size)
    )

    if cursor is not None:
        return query.with_after(cursor).do()
    else:
        return query.do()
```
The rest of the tutorial will be inside of a for loop.
This allows us to create an Atlas Map for all of the classes in the database. 
```python
for c, p in zip(classes, props):
```
!!! note "Map out only one class"
    If you would like to map only a single class set `c` equal to the class name and `p` equal to a list with the class properties

We will now create an Atlas Project which will eventually contain all of our embeddings and data
```python
project = AtlasProject(
    name=c,
    unique_id_field="id",
    modality="embedding",
)
```
Now we use a while loop to access all of the data from each class, which we do in batches using our helper function, in this case we have a batch size of 10,000. We break the while loop when a call to the helper function returns no values. 

We then set our cursor to the id of the datapoint we left off at, and append the vectors to a list, which we then convert into a numpy array. 
!!! note "To Not Include Properties" 
    To not include a property add the property name to the list titled `not_data`. If it the property is an additional property add the property name to `un_data`
We then parse our data only including the properties we want. 
Finally we add the embeddings to our atlas project along with our parsed data. 
```python
cursor = None
while True:
    response = get_batch_with_cursor(client, c, p, 10000, cursor)
    if len(response["data"]["Get"][c]) == 0:
        break
    cursor = response["data"]["Get"][c][-1]["_additional"]["id"]
    vectors = []
    for i in response["data"]["Get"][c]:
        vectors.append(i["_additional"]["vector"])

    embeddings = np.array(vectors)
    data = []
    not_data = ["_additional"]
    un_data = ["vector"]
    for i in response["data"]["Get"][c]:
        j = {key: value for key, value in i.items() if key not in not_data}
        k = {
            key: value
            for key, value in i["_additional"].items()
            if key not in un_data
        }
        j = j | k
        data.append(j)
    with project.wait_for_project_lock():
        project.add_embeddings(
            embeddings=embeddings,
            data=data,
        )
```

Finally we will build our map with the given parameters using `create_index()`
!!! note "Add Topic Labels"
    If you want labels on your atlas map add the following line of code using the property name that you want to build the labels for: `topic_label_field= "PROPERTY NAME"`
```python
project.create_index(
    name=c,
    colorable_fields=p,
    build_topic_model=True,
)
```

You can find the source code [here](https://github.com/nomic-ai/maps/blob/main/maps/weaviate_script.py)

## Pinecone

First, find your Pinecone and Atlas API keys.
```python
import pinecone
import numpy as np
from nomic import atlas
import nomic
pinecone.init(api_key='YOUR PINECONE API KEY', environment='us-east1-gcp')
nomic.login('YOUR NOMIC API KEY')
```

Below we will create an example Pinecone Vector Database Index and fill it with 1000 random embeddings. 
!!! note "Use your own index"
    If you have an existing Pinecone Index, you can skip this step and just import the Index as usual.
```python
pinecone.create_index("quickstart", dimension=128, metric="euclidean", pod_type="p1")

index = pinecone.Index("quickstart")

num_embeddings = 1000
embeddings_for_pinecone = np.random.rand(num_embeddings, 128)
index.upsert([(str(i), embeddings_for_pinecone[i].tolist()) for i in range(num_embeddings)])
```

Next, you'll need to get the ID's of all of your embeddings to extract them from your Pinecone Index. In our previous example, we just used the integers 0-999 as our ID's. Then, extract the embeddings out into a numpy array. Once you have embeddings, send them over to Atlas. 

```python
vectors = index.fetch(ids=[str(i) for i in range(num_embeddings)])

ids = []
embeddings = []
for id, vector in vectors['vectors'].items():
    ids.append(id)
    embeddings.append(vector['values'])

embeddings = np.array(embeddings)

atlas.map_embeddings(embeddings=embeddings, data=[{'id': id} for id in ids], id_field='id')
```

You can find the full source code [here](https://github.com/nomic-ai/maps/blob/main/maps/pinecone_index.py)
