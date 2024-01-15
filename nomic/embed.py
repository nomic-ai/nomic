import base64
import os
from io import BytesIO
from typing import List, Union
import concurrent
import concurrent.futures

import PIL
import PIL.Image
import requests

from .dataset import AtlasClass
from .settings import *

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


    def run_inference(batch):
        response = requests.post(
            atlas_class.atlas_api_path + "/v1/embedding/image",
            headers=atlas_class.header,
            data={"model": "nomic-embed-vision-v1"},
            files=batch,
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(response.text)
            raise Exception(response.status_code)

    def resize_pil(img):
        width, height = img.size
        #if image is too large, downsample before sending over the wire
        max_width = 512
        max_height = 512
        if max_width > 512 or max_height > 512:
            downsize_factor = max(width/max_width, height/max_height)
            img.resize((width/downsize_factor, height/downsize_factor))
        return img

    def send_request(i):
        image_batch = []
        shard = images[i:i+IMAGE_EMBEDDING_BATCH_SIZE]
        # process images into base64 encoded strings (for now)
        for image in shard:
            # TODO implement check for bytes.
            # TODO implement check for a valid image.
            if isinstance(image, str) and os.path.exists(image):
                img = Image.open(image)
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                image_batch.append(('images', buffered.getvalue()))

            elif isinstance(image, PIL.Image.Image):
                img = resize_pil(image)
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                image_batch.append(('images', buffered.getvalue()))
            else:
                raise ValueError(f"Not a valid file: {image}")
        response = run_inference(batch=image_batch)
        print(response['usage'])
        return (i, response)


    # naive batching, we should parallelize this across threads like we do with uploads.
    responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(send_request, i): i for i in range(0, len(images), IMAGE_EMBEDDING_BATCH_SIZE)}
        while futures:
            done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            # process any completed futures
            for future in done:
                response = future.result()
                responses.append(response)
                del futures[future]

        responses = sorted(responses, key=lambda x: x[0])
        responses = [e[1] for e in responses]

        final_response = {}
        final_response['embeddings'] = [embedding for response in responses for embedding in response['embeddings']]
        final_response['usage'] = {}
        final_response['usage']['prompt_tokens'] = sum([response['usage']['prompt_tokens'] for response in responses])
        final_response['usage']['total_tokens'] = final_response['usage']['prompt_tokens']

    return final_response
