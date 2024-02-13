import logging
import time
from typing import List, Union

import requests
import numpy as np

from .dataset import AtlasClass
from .settings import *

atlas_class = None

MAX_TEXT_REQUEST_SIZE = 50
MIN_EMBEDDING_DIMENSIONALITY = 128

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


def text_api_request(texts: List[str], model: str, task_type: str, dimensionality: int = None):
    global atlas_class
    response = request_backoff(
        lambda: requests.post(
            atlas_class.atlas_api_path + "/v1/embedding/text",
            headers=atlas_class.header,
            json={"texts": texts, "model": model, "task_type": task_type, "dimensionality": dimensionality},
        )
    )

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception((response.status_code, response.text))


def embeddings(embeddings: np.array):
    """
    Reduces embeddings to 2D

    """
    global atlas_class
    if atlas_class is None:
        atlas_class = AtlasClass()


    response = requests.post(
        atlas_class.atlas_api_path + "/v1/reduce/",
        headers=atlas_class.header,
        json={"embeddings": embeddings.tolist() , "model": 'nomic-project-v1', "n_neighbors": 15},
    )

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception((response.status_code, response.text))

