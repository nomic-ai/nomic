from __future__ import annotations

import concurrent
import concurrent.futures
import logging
import os
import time
from io import BytesIO
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Sequence, Tuple, Union, overload
from urllib.parse import urlparse

import PIL
import PIL.Image
import requests

from .dataset import AtlasClass
from .settings import *

try:
    from gpt4all import CancellationError, Embed4All
except ImportError:
    if TYPE_CHECKING:
        raise
    Embed4All = None

atlas_class: Optional[AtlasClass] = None

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
    response = callable()
    for attempt in range(max_retries + 1):
        if attempt == max_retries:
            return response
        if backoff_if(response.status_code):
            delay = init_backoff * (ratio**attempt)
            logging.info(f"server error, backing off for {int(delay)}s")
            time.sleep(delay)
            response = callable()
        else:
            break
    return response


def text_api_request(
    texts: List[str], model: str, task_type: str, dimensionality: Optional[int] = None, long_text_mode: str = "truncate"
):
    global atlas_class

    assert atlas_class is not None
    text_api_url = atlas_class.atlas_api_path + "/v1/embedding/text"
    text_api_header = atlas_class.header

    response = request_backoff(
        lambda: requests.post(
            text_api_url,
            headers=text_api_header,
            json={
                "texts": texts,
                "model": model,
                "task_type": task_type,
                "dimensionality": dimensionality,
                "long_text_mode": long_text_mode,
            },
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
    inference_mode: Literal["remote"] = ...,
) -> dict[str, Any]: ...


@overload
def text(
    texts: list[str],
    *,
    model: str = ...,
    task_type: str = ...,
    dimensionality: int | None = ...,
    long_text_mode: str = ...,
    inference_mode: Literal["local", "dynamic"],
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
    model: str = "nomic-embed-text-v1.5",
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
        inference_mode: How to generate embeddings. One of `remote`, `local` (Embed4All), or `dynamic` (automatic).
            Defaults to `remote`.
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

    use_embed4all = dynamic_mode = False
    if inference_mode == "remote":
        pass
    elif inference_mode == "local":
        use_embed4all = True
    elif inference_mode == "dynamic":
        use_embed4all = dynamic_mode = True
    else:
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
        try:
            return _text_embed4all(
                texts,
                model,
                task_type,
                dimensionality,
                long_text_mode,
                dynamic_mode,
                device=device,
                **kwargs,
            )
        except CancellationError:
            pass  # dynamic mode chose to use Atlas, fall through

    return _text_atlas(texts, model, task_type, dimensionality, long_text_mode)


def _text_atlas(
    texts: list[str],
    model: str,
    task_type: str,
    dimensionality: int | None,
    long_text_mode: str,
) -> dict[str, Any]:
    global atlas_class
    assert task_type in [
        "search_query",
        "search_document",
        "classification",
        "clustering",
    ], f"Invalid task type: {task_type}"

    if dimensionality and dimensionality < MIN_EMBEDDING_DIMENSIONALITY:
        logging.warning(
            f"Dimensionality {dimensionality} is less than the suggested of {MIN_EMBEDDING_DIMENSIONALITY}. Performance may be degraded."
        )

    if atlas_class is None:
        atlas_class = AtlasClass()
    max_workers = 10
    chunksize = MAX_TEXT_REQUEST_SIZE
    smallchunk = max(1, int(len(texts) / max_workers))
    # if there are fewer texts per worker than the max chunksize just split them evenly
    chunksize = min(smallchunk, chunksize)

    combined = {"embeddings": [], "usage": {}, "model": model, "inference_mode": "remote"}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for chunkstart in range(0, len(texts), chunksize):
            chunkend = min(len(texts), chunkstart + chunksize)
            chunk = texts[chunkstart:chunkend]
            futures.append(executor.submit(text_api_request, chunk, model, task_type, dimensionality, long_text_mode))

        for future in futures:
            response = future.result()
            assert response["model"] == model
            combined["embeddings"] += response["embeddings"]
            for counter, value in response["usage"].items():
                combined["usage"][counter] = combined["usage"].get(counter, 0) + value
    return combined


_embed4all: Optional[Embed4All] = None
_embed4all_kwargs: Optional[dict[str, Any]] = None


def _text_embed4all(
    texts: list[str],
    model: str,
    task_type: str,
    dimensionality: Optional[int],
    long_text_mode: str,
    dynamic_mode: bool,
    **kwargs: Any,
) -> dict[str, Any]:
    global _embed4all, _embed4all_kwargs

    try:
        g4a_model = _EMBED4ALL_MODELS[model]
    except KeyError:
        raise ValueError(f"Unsupported model for local embeddings: {model!r}") from None

    if not texts:
        # special-case this since Embed4All doesn't allow it
        return {"embeddings": [], "usage": {}, "model": model, "inference_mode": "local"}

    if _embed4all is None or _embed4all.gpt4all.config["filename"] != g4a_model or _embed4all_kwargs != kwargs:
        if _embed4all is not None:
            _embed4all.close()
        _embed4all = Embed4All(g4a_model, **kwargs)
        _embed4all_kwargs = kwargs

    def cancel_cb(batch_sizes: list[int], backend: str) -> bool:
        # TODO(jared): make this more accurate
        n_tokens = sum(batch_sizes)
        limits = {"cpu": 16, "kompute": 32, "metal": 1024}
        return n_tokens > limits[backend]

    output = _embed4all.embed(
        texts,
        prefix=task_type,
        dimensionality=dimensionality,
        long_text_mode=long_text_mode,
        return_dict=True,
        atlas=True,
        cancel_cb=cancel_cb if dynamic_mode else None,
    )
    ntok = output["n_prompt_tokens"]
    usage = {"prompt_tokens": ntok, "total_tokens": ntok}
    return {"embeddings": output["embeddings"], "usage": usage, "model": model, "inference_mode": "local"}


def free_embedding_model() -> None:
    """Free the current Embed4All instance and its associated system resources."""
    global _embed4all, _embed4all_kwargs
    if _embed4all is not None:
        _embed4all.close()
        _embed4all = _embed4all_kwargs = None


def image_api_request(
    images: Optional[List[Tuple[str, bytes]]] = None,
    urls: Optional[List[str]] = None,
    model: str = "nomic-embed-vision-v1.5",
):
    global atlas_class

    assert atlas_class is not None
    atlas_url = atlas_class.atlas_api_path
    atlas_header = atlas_class.header

    response = request_backoff(
        lambda: requests.post(
            atlas_url + "/v1/embedding/image",
            headers=atlas_header,
            data={"model": model, "urls": urls},
            files=images,
        )
    )

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception((response.status_code, response.text))


def resize_pil(img):
    width, height = img.size
    # if image is too large, downsample before sending over the wire
    max_width = 512
    max_height = 512
    if width > max_width or height > max_height:
        downsize_factor = max(width // max_width, height // max_height)
        img = img.resize((width // downsize_factor, height // downsize_factor))
    return img


def _is_valid_url(url):
    if not isinstance(url, str):
        return False
    parsed_url = urlparse(url)
    return all([parsed_url.scheme, parsed_url.netloc])


def image(images: Sequence[Union[str, PIL.Image.Image]], model: str = "nomic-embed-vision-v1.5"):
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

    if isinstance(images, str):
        raise TypeError("'images' parameter must be list of strings or PIL images, not str")

    urls = []
    image_batch = []
    for image in images:

        if isinstance(image, str):
            if os.path.exists(image):
                img = resize_pil(PIL.Image.open(image)).convert("RGB")
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                image_batch.append(("images", buffered.getvalue()))
            elif _is_valid_url(image):
                # Send URL as data
                urls.append(image)
            else:
                raise ValueError(f"Invalid image path or url: {image}")

        elif isinstance(image, PIL.Image.Image):
            img = resize_pil(image)
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            image_batch.append(("images", buffered.getvalue()))
        else:
            raise ValueError(f"Not a valid file: {image}")

    if len(urls) > 0 and len(image_batch) > 0:
        raise ValueError("Provide either urls or image files/objects, not both.")

    num_images = len(urls) + len(image_batch)
    combined = {"embeddings": [], "usage": {}, "model": model}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for chunkstart in range(0, num_images, chunksize):
            chunkend = min(num_images, chunkstart + chunksize)
            image_chunk = None
            url_chunk = None
            if len(image_batch) > 0:
                image_chunk = image_batch[chunkstart:chunkend]
            else:
                url_chunk = urls[chunkstart:chunkend]
            futures.append(executor.submit(image_api_request, image_chunk, url_chunk, model))

        for future in futures:
            response = future.result()
            assert response["model"] == model
            combined["embeddings"] += response["embeddings"]
            for counter, value in response["usage"].items():
                combined["usage"][counter] = combined["usage"].get(counter, 0) + value
    return combined
