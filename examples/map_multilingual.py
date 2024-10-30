from nomic import atlas
from datasets import load_dataset

dataset = load_dataset("bbaaaa/iwslt14-de-en", split="train")

max_documents = 50_000
selected = dataset[:max_documents]["translation"]

documents = []
for doc in selected:
    en_data = {"text": doc["en"], "en": doc["en"], "de": doc["de"], "language": "en"}
    de_data = {"text": doc["de"], "en": doc["en"], "de": doc["de"], "language": "de"}
    documents.append(en_data)
    documents.append(de_data)
project = atlas.map_data(data=documents,
                          indexed_field='text',
                          identifier='English-German 50k Translations',
                          description='50k Examples from the iwslt14-de-en dataset hosted on huggingface.',
                          embedding_model='gte-multilingual-base',
                          )
print(project.maps)

