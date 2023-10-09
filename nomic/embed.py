import PIL.Image
import requests
import os
import base64
from io import BytesIO

from typing import List, Union
import PIL
from .project import AtlasClass


atlas_class = AtlasClass()



def text(texts: List[Union[str, PIL.Image.Image]], model: str = 'nomic-embed-text-v1'):
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

def images(images: List[str], model: str = 'nomic-embed-image-v1'):
    """
    Generates embeddings for the given images.

    Args:
        images: the images to embed. Can be file paths to images, image-file bytes or Pillow objects
        model: the model to use when embedding

    Returns:
        An object containing your embeddings and request metadata
    """

    image_batch = []

    for image in images:
        #TODO implement check for bytes.
        #TODO implement check for pillow image.
        #TODO implement check for a valid image.
        if isinstance(image, str) and os.path.exists(image):
            with open(image, "rb") as image_file:
                base64_image_string = base64.b64encode(image_file.read()).decode('utf-8')
                image_batch.append(base64_image_string)
        elif isinstance(image, PIL.Image.Image):
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            image_batch.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
        else:
            raise ValueError(f"Not a valid file: {image}")


    response = requests.post(
        atlas_class.atlas_api_path + "/v1/embedding/image",
        headers=atlas_class.header,
        json={'images': image_batch, 'model': model},
    )

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(str(response.json()))



