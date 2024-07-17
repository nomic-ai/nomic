import numpy as np
from nomic import embed
from typing import Union, List
from datasets import load_dataset
from PIL import Image


class SemanticSearchClassifier:
    def __init__(self, threshold: float, text_query: str):
        self.threshold = threshold

        text_emb = embed.text([text_query], task_type="search_query")["embeddings"]
        self.text_emb: np.ndarray = np.array(text_emb, dtype=np.float16)

        
    def predict(self, images: List[Union[str, Image.Image]]) -> bool:
        image_emb = embed.image(images)["embeddings"]
        image_emb = np.array(image_emb, dtype=np.float16)

        similarity = np.dot(self.text_emb, image_emb.T)
        return np.squeeze(similarity > self.threshold).tolist()

print(f"Building classifier")        
classifier = SemanticSearchClassifier(threshold=0.058, text_query="a tiny white ball")

print(f"Loading dataset")
dataset = load_dataset('frgfm/imagenette', '160px')['train']

# first three should return True, last two should return False
ids = [7450, 6828, 7464, 2343, 3356]
images = [dataset[i]["image"] for i in ids]

print(f"Predicting")
predictions = classifier.predict(images)
assert predictions == [True, True, True, False, False], f"Predictions should be [True, True, True, False, False], got {predictions=}"
for i, pred in enumerate(predictions):
    print(f"Prediction for image {ids[i]}: {pred}")

    
label_mapping = {0: "baseball", 1: "cricket", 2: "hockey"}
sport_dataset = load_dataset('nihaludeen/sports_classification', split="train")
sport_images = sport_dataset["image"]
sport_labels = sport_dataset["label"]
predictions = classifier.predict(sport_images)
positive_predictions = [i for i, pred in enumerate(predictions) if pred]
sport_images[positive_predictions[0]].show() 