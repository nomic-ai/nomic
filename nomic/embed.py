from __future__ import annotations

import concurrent
import concurrent.futures
import logging
import os
import time
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union, overload

import PIL
import PIL.Image
import requests

from .dataset import AtlasClass
from .settings import *

try:
    from gpt4all import Embed4All
except ImportError:
    if not TYPE_CHECKING:
        Embed4All = None

atlas_class = None

MAX_TEXT_REQUEST_SIZE = 50
MAX_IMAGE_REQUEST_SIZE = 512
MIN_EMBEDDING_DIMENSIONALITY = 64

# mapping of Atlas model name -> Embed4All model filename
_EMBED4ALL_MODELS = {
    "nomic-embed-text-v1": "nomic-embed-text-v1.f16.gguf",
    "nomic-embed-text-v1.5": "nomic-embed-text-v1.5.f16.gguf",
}


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


@overload
def text(
    texts: list[str],
    *,
    model: str = ...,
    task_type: str = ...,
    dimensionality: int | None = ...,
    long_text_mode: str = ...,
    inference_mode: Literal["atlas"] = ...,
) -> dict[str, Any]: ...
@overload
def text(
    texts: list[str],
    *,
    model: str = ...,
    task_type: str = ...,
    dimensionality: int | None = ...,
    long_text_mode: str = ...,
    inference_mode: Literal["local"],
    device: str | None = ...,
    **kwargs: Any,
) -> dict[str, Any]: ...
@overload
def text(
    texts: list[str],
    *,
    model: str = ...,
    task_type: str = ...,
    dimensionality: int | None = ...,
    long_text_mode: str = ...,
    inference_mode: str,
    device: str | None = ...,
    **kwargs: Any,
) -> dict[str, Any]: ...


def text(
    texts: list[str],
    *,
    model: str = "nomic-embed-text-v1",
    task_type: str = "search_document",
    dimensionality: int | None = None,
    long_text_mode: str = "truncate",
    inference_mode: str = "remote",
    device: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Generates embeddings for the given text.

    Args:
        texts: The text to embed.
        model: The model to use when embedding.
        task_type: The task type to use when embedding. One of `search_query`, `search_document`, `classification`, `clustering`.
        dimensionality: The embedding dimension, for use with Matryoshka-capable models. Defaults to full-size.
        long_text_mode: How to handle texts longer than the model can accept. One of `mean` or `truncate`.
        inference_mode: How to generate embeddings. One of `remote` or `local` (Embed4All). Defaults to `remote`.
        device: The device to use for local embeddings. Defaults to CPU, or Metal on Apple Silicon. It can be set to:
            - "gpu": Use the best available GPU.
            - "amd", "nvidia": Use the best available GPU from the specified vendor.
			- A specific device name from the output of `GPT4All.list_gpus()`
        kwargs: Remaining arguments are passed to the Embed4All contructor.

    Returns:
        A dict containing your embeddings and request metadata
    """
    if isinstance(texts, str):
        raise TypeError("'texts' parameter must be list[str], not str")

    modes = {
        "remote": False,
        "local": True,
    }

    try:
        use_embed4all = modes[inference_mode]
    except KeyError:
        raise ValueError(f"Unknown inference mode: {inference_mode!r}") from None

    if inference_mode == "remote":
        if device is not None:
            raise TypeError(f"device argument cannot be used with inference_mode='remote'")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")
    elif Embed4All is None:
        raise RuntimeError(
            f"The 'gpt4all' package is required for local inference. Suggestion: `pip install \"nomic[local]\"`",
        )

    if use_embed4all:
        return _text_embed4all(texts, model, task_type, dimensionality, long_text_mode, **kwargs)

    return _text_atlas(texts, model, task_type, dimensionality, long_text_mode)


def _text_atlas(
    texts: list[str],
    model: str,
    task_type: str,
    dimensionality: int | None,
    long_text_mode: str,
) -> dict[str, Any]:
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

    combined = {'embeddings': [], 'usage': {}, 'model': model}
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


_embed4all: Embed4All | None = None
_embed4all_kwargs: dict[str, Any] | None = None


def _text_embed4all(
    texts: list[str],
    model: str,
    task_type: str,
    dimensionality: int | None,
    long_text_mode: str,
    **kwargs: Any,
) -> dict[str, Any]:
    global _embed4all, _embed4all_kwargs

    try:
        g4a_model = _EMBED4ALL_MODELS[model]
    except KeyError:
        raise ValueError(f"Unsupported model for local embeddings: {model!r}") from None

    if _embed4all is None or _embed4all.gpt4all.config["filename"] != g4a_model or _embed4all_kwargs != kwargs:
        _embed4all = Embed4All(g4a_model, **kwargs)
        _embed4all_kwargs = kwargs

    output = _embed4all.embed(
        texts,
        prefix=task_type,
        dimensionality=dimensionality,
        long_text_mode=long_text_mode,
        return_dict=True,
        atlas=True,
    )
    ntok = output["n_prompt_tokens"]
    usage = {"prompt_tokens": ntok, "total_tokens": ntok}
    return {"embeddings": output["embeddings"], "usage": usage, "model": model}


def free_embedding_model() -> None:
    """Free the current Embed4All instance and its associated system resources."""
    global _embed4all, _embed4all_kwargs
    if _embed4all is not None:
        _embed4all.close()
        _embed4all = _embed4all_kwargs = None


def image_api_request(images: List[Tuple[str, bytes]], model: str = 'nomic-embed-vision-v1'):
    global atlas_class
    response = request_backoff(
        lambda: requests.post(
            atlas_class.atlas_api_path + "/v1/embedding/image",
            headers=atlas_class.header,
            data={"model": model},
            files=images,
        )
    )

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception((response.status_code, response.text))
    
    
def resize_pil(img):
    width, height = img.size
    #if image is too large, downsample before sending over the wire
    max_width = 512
    max_height = 512
    if max_width > 512 or max_height > 512:
        downsize_factor = max(width/max_width, height/max_height)
        img.resize((width/downsize_factor, height/downsize_factor))
    return img


def images(images: Iterable[Union[str, PIL.Image.Image]], model: str = 'nomic-embed-vision-v1'):
    """
    Generates embeddings for the given images.

    Args:
        images: the images to embed. Can be file paths to images, image-file bytes or Pillow objects
        model: the model to use when embedding

    Returns:
        An object containing your embeddings and request metadata
    """
    global atlas_class

    if atlas_class is None:
        atlas_class = AtlasClass()

    max_workers = 10
    chunksize = MAX_IMAGE_REQUEST_SIZE
    smallchunk = max(1, int(len(images) / max_workers))
    # if there are fewer images per worker than the max chunksize just split them evenly
    chunksize = min(smallchunk, chunksize)

    image_batch = []
    for image in images:
        if isinstance(image, str) and os.path.exists(image):
                img = resize_pil(PIL.Image.open(image))
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                image_batch.append(("images", buffered.getvalue()))

        elif isinstance(image, PIL.Image.Image):
            img = resize_pil(image)
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            image_batch.append(("images", buffered.getvalue()))
        else:
            raise ValueError(f"Not a valid file: {image}")


    combined = {'embeddings': [], 'usage': {}, 'model': model}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for chunkstart in range(0, len(image_batch), chunksize):
            chunkend = min(len(image_batch), chunkstart + chunksize)
            chunk = image_batch[chunkstart:chunkend]
            futures.append(executor.submit(image_api_request, chunk, model))

        for future in futures:
            response = future.result()
            assert response['model'] == model
            combined['embeddings'] += response['embeddings']
            for counter, value in response['usage'].items():
                combined['usage'][counter] = combined['usage'].get(counter, 0) + value
    return combined
