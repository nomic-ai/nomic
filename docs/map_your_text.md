# Map Your Text
Map your text documents with Atlas using the `map_text` function.
Atlas will ingest your documents, organize them with state-of-the-art AI and then serve you back an interactive map.
Any interaction you do with your data (e.g. tagging) can be accessed programmatically with the Atlas Python API.

## Map text with Atlas
When sending text you should specify an `indexed_field` in the `map_text` function. This lets Atlas know what metadata field to use when building your map.

=== "Atlas Embed"

    ``` py title="map_text_with_atlas.py"
    from nomic import atlas
    import numpy as np
    from datasets import load_dataset
    
    #Make a dataset with the shape [{'col1': 'val', 'col2': 'val', ...}, etc]
    #Tip: if you're working with a pandas dataframe
    #     use pandas.DataFrame.to_dict('records')

    dataset = load_dataset('ag_news')['train']
    
    max_documents = 10000
    subset_idxs = np.random.randint(len(dataset), size=max_documents).tolist()
    documents = [dataset[i] for i in subset_idxs]
    
    project = atlas.map_text(data=documents,
                              indexed_field='text',
                              name='News 10k Example',
                              colorable_fields=['label'],
                              description='News 10k Example.'
                              )
    ```

=== "Output"

    ``` bash
    https://atlas.nomic.ai/map/0642e9a1-12d9-4504-a987-9ca50ecd5327/699afdee-cea0-4805-9c84-12eca6dbebf8
    ```


## Map text with your own models
Nomic integrates with embedding providers such as [co:here](https://cohere.ai/) and [huggingface](https://huggingface.co/models) to help you build maps of text.


### Text maps with a 🤗 HuggingFace model
This code snippet is a complete example of how to make a map with a HuggingFace model.
[Example Huggingface Map](https://atlas.nomic.ai/map/60e57e91-c573-4d1f-85ac-2f00f2a075ae/f5bf58cf-f40b-439d-bd0d-d3a4a8b98496)

!!! note

    This example requires additional packages. Install them with
    ```bash
    pip install datasets transformers torch
    ```
=== "HuggingFace Example"

    ``` py title="map_with_huggingface.py"
    from nomic import atlas
    from transformers import AutoTokenizer, AutoModel
    import numpy as np
    import torch
    from datasets import load_dataset
    
    #make dataset
    max_documents = 10000
    dataset = load_dataset("sentiment140")['train']
    documents = [dataset[i] for i in np.random.randint(len(dataset), size=max_documents).tolist()]
    
    
    model = AutoModel.from_pretrained("prajjwal1/bert-mini")
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-mini")
    
    embeddings = []
    
    with torch.no_grad():
        batch_size = 10 # lower this if needed
        for i in range(0, len(documents), batch_size):
            batch = [document['text'] for document in documents[i:i+batch_size]]
            encoded_input = tokenizer(batch, return_tensors='pt', padding=True)
            cls_embeddings = model(**encoded_input)['last_hidden_state'][:, 0]
            embeddings.append(cls_embeddings)
    
    embeddings = torch.cat(embeddings).numpy()
    
    response = atlas.map_embeddings(embeddings=embeddings,
                                    data=documents,
                                    colorable_fields=['sentiment'],
                                    name="Huggingface Model Example",
                                    description="An example of building a text map with a huggingface model.")
    
    print(response)
    ```

=== "Output"

    ``` bash
    https://atlas.nomic.ai/map/60e57e91-c573-4d1f-85ac-2f00f2a075ae/f5bf58cf-f40b-439d-bd0d-d3a4a8b98496
    ```


### Text maps with a Cohere model

Obtain an API key from [cohere.ai](https://os.cohere.ai) to embed your text data.

Add your Cohere API key to the below example to see how their large language model organizes text from a sentiment analysis dataset.

[Sentiment Analysis Map](https://atlas.nomic.ai/map/63b3d891-f807-44c5-abdf-2a95dad05b41/db0fa89e-6589-4a82-884b-f58bfb60d641)

!!! note

    This example requires additional packages. Install them with
    ```bash
    pip install datasets
    ```

=== "Co:here Example"

    ``` py title="map_hf_dataset_with_cohere.py"
    from nomic import atlas
    from nomic import CohereEmbedder
    import numpy as np
    from datasets import load_dataset

    cohere_api_key = ''
    
    dataset = load_dataset("sentiment140")['train']
    
    max_documents = 10000
    subset_idxs = np.random.randint(len(dataset), size=max_documents).tolist()
    documents = [dataset[i] for i in subset_idxs]

    embedder = CohereEmbedder(cohere_api_key=cohere_api_key)
    
    print(f"Embedding {len(documents)} documents with Cohere API")
    embeddings = embedder.embed(texts=[document['user'] for document in documents],
                                model='small')
    
    if len(embeddings) != len(documents):
        raise Exception("Embedding job failed")
    print("Embedding job complete.")
    
    response = atlas.map_embeddings(embeddings=np.array(embeddings),
                                    data=documents,
                                    colorable_fields=['sentiment'],
                                    name='Sentiment 140',
                                    description='A 10,000 point sample of the huggingface sentiment140 dataset embedded with the co:here small model.',
                                    )
    print(response)
    ```

=== "Output"

    ``` bash
    https://atlas.nomic.ai/map/ff2f89df-451e-49c4-b7a3-a608d7375961/f433cbd1-e728-49da-8c83-685cd613788b
    ```
