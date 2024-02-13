import base64
import concurrent
import concurrent.futures
import logging
import os
import time
from io import BytesIO
from typing import List, Union

import PIL
import PIL.Image
import requests

from .dataset import AtlasClass
from .settings import *

atlas_class = None

MAX_TEXT_REQUEST_SIZE = 50
MIN_EMBEDDING_DIMENSIONALITY = 64

def is_backoff_status_code(code: int):
    if code == 429 or code >= 500:
        # server error, do backoff
        return True
    return False


def request_backoff(
    callable,
    init_backoff=1.0,
    ratio=2.0,
    max_retries=5,
    backoff_if=is_backoff_status_code,
):
    for attempt in range(max_retries + 1):
        response = callable()
        if attempt == max_retries:
            return response
        if backoff_if(response.status_code):
            delay = init_backoff * (ratio**attempt)
            logging.info(f"server error, backing off for {int(delay)}s")
            time.sleep(delay)
        else:
            return response


def text_api_request(texts: List[str], model: str, task_type: str, dimensionality: int = None, long_text_mode: str = "truncate"):
    global atlas_class
    response = request_backoff(
        lambda: requests.post(
            atlas_class.atlas_api_path + "/v1/embedding/text",
            headers=atlas_class.header,
            json={"texts": texts, "model": model, "task_type": task_type, "dimensionality": dimensionality, "long_text_mode": long_text_mode},
        )
    )

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception((response.status_code, response.text))


def text(texts: List[str], model: str = "nomic-embed-text-v1", task_type: str = "search_document", dimensionality: int = None, long_text_mode: str = "truncate"):
    """
    Generates embeddings for the given text.

    Args:
        texts: the text to embed
        model: the model to use when embedding
        task_type: the task type to use when embedding. One of `search_query`, `search_document`, `classification`, `clustering`

    Returns:
        An object containing your embeddings and request metadata
    """
    global atlas_class
    assert task_type in ["search_query", "search_document", "classification", "clustering"], f"Invalid task type: {task_type}"

    if dimensionality and dimensionality < MIN_EMBEDDING_DIMENSIONALITY:
        logging.warning(f"Dimensionality {dimensionality} is less than the suggested of {MIN_EMBEDDING_DIMENSIONALITY}. Performance may be degraded.")

    if atlas_class is None:
        atlas_class = AtlasClass()
    max_workers = 10
    chunksize = MAX_TEXT_REQUEST_SIZE
    smallchunk = max(1, int(len(texts) / max_workers))
    # if there are fewer texts per worker than the max chunksize just split them evenly
    chunksize = min(smallchunk, chunksize)

    combined = {'embeddings': [], 'usage': {}, 'model': 'model'}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for chunkstart in range(0, len(texts), chunksize):
            chunkend = min(len(texts), chunkstart + chunksize)
            chunk = texts[chunkstart:chunkend]
            futures.append(executor.submit(text_api_request, chunk, model, task_type, dimensionality, long_text_mode))

        for future in futures:
            response = future.result()
            assert response['model'] == model
            combined['embeddings'] += response['embeddings']
            for counter, value in response['usage'].items():
                combined['usage'][counter] = combined['usage'].get(counter, 0) + value
    return combined


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
                img = PIL.Image.open(image)
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
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
