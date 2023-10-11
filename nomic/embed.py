import PIL.Image
import requests
import os
import base64
from io import BytesIO

from typing import List, Union
import PIL
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

def images(images: Union[str, PIL.Image.Image], model: str = 'nomic-embed-vision-v1'):
    """
    Generates embeddings for the given images.

    Args:
        images: the images to embed. Can be file paths to images, image-file bytes or Pillow objects
        model: the model to use when embedding

    Returns:
        An object containing your embeddings and request metadata
    """

    batch_size = 250

    def run_inference(batch):
        response = requests.post(
            atlas_class.atlas_api_path + "/v1/embedding/image",
            headers=atlas_class.header,
            data={"model": "nomic-embed-vision-v1"},
            files=batch
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(response.text)
            raise Exception(response.status_code)


    #naive batching, we should parallelize this across threads like we do with uploads.
    #TODO this should all be re-written prior to public release
    responses = []
    for i in range(0, len(images), batch_size):
        image_batch = []

        # process images into base64 encoded strings (for now)
        for image in images[i:i + batch_size]:
            #TODO implement check for bytes.
            #TODO implement check for a valid image.
            if isinstance(image, str) and os.path.exists(image):
                image_batch.append(('images', open(image, "rb")))

            elif isinstance(image, PIL.Image.Image):
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                image_batch.append(('images', buffered.getvalue()))
            else:
                raise ValueError(f"Not a valid file: {image}")
        print(len(image_batch))
        response = run_inference(batch=image_batch)
        print(response['usage'])
        responses.append(response)

    final_response = {}
    final_response['embeddings'] = [embedding for response in responses for embedding in response['embeddings']]
    final_response['usage'] = {}
    final_response['usage']['prompt_tokens'] = sum([response['usage']['prompt_tokens'] for response in responses])
    final_response['usage']['total_tokens'] = final_response['usage']['prompt_tokens']

    return final_response









