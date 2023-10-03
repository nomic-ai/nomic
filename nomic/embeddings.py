import requests
from typing import List
from .project import AtlasClass


atlas_class = AtlasClass()



def text(texts: List[str], model: str = 'nomic-embed-text-v1'):
    """
    Generates embeddings for the given text.

    Args:
        texts: the text to embed
        model: the model to use when embedding

    Returns:
        An object containing your embeddings and request metadata
    """

    response = requests.post(
        atlas_class.atlas_api_path + "/v1/embedding/text",
        headers=atlas_class.header,
        json={'texts': texts, 'model': model},
    )

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(str(response.json()))



