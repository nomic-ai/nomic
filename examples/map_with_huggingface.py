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
        cls_embeddings = model(**encoded_input)['last_hidden_state'][:, 0] #
        embeddings.append(cls_embeddings)

embeddings = torch.cat(embeddings).numpy()
print(embeddings.shape)

response = atlas.map_embeddings(embeddings=embeddings,
                                data=documents,
                                colorable_fields=['sentiment'],
                                name="Huggingface Model Example",
                                description="An example of building a text map with a huggingface model.")

print(response)


