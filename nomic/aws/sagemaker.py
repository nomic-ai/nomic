# Note: This file contains methods to prepare embedding requests for Sagemaker.
# It may make sense to move code eventually to embed.py or somewhere more generic but
# it currently lives here to separate out dependencies.

import hashlib
import logging
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np
import tritonclient.http.aio as aiohttpclient
from tokenizers import Tokenizer
from tritonclient.http import InferenceServerClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


text_embedding_model_info = {
    "nomic-embed-text-v1.5": {
        "dim": 768,
        "max_length": 2048,
        "pad_id": 0,
        "recommended_dims": [768, 512, 384, 256, 128],
    },
}

NULL_PLACEHOLDER = hashlib.md5(b"nomic null").hexdigest()
EMPTY_PLACEHOLDER = hashlib.md5(b"nomic empty").hexdigest()


class NomicTextEmbeddingModel(Enum):
    nomic_embed_text_v1_5 = "nomic-embed-text-v1.5"

    def hamming_capable(self):
        return self == NomicTextEmbeddingModel.nomic_embed_text_v1_5

    def matryoshka_capable(self):
        return self == NomicTextEmbeddingModel.nomic_embed_text_v1_5

    def dim(self):
        return text_embedding_model_info[self.value]["dim"]

    def recommended_dims(self):
        return text_embedding_model_info[self.value].get("recommended_dims") or [
            self.dim()
        ]

    def max_length(self):
        return text_embedding_model_info[self.value]["max_length"]

    def pad_token(self) -> int:
        return text_embedding_model_info[self.value]["pad_id"]


def null_empty_placeholder(text: Optional[str]) -> str:
    """
    If text is null or empty replace with a placeholder nonsense string.
    """
    if text is None:
        return NULL_PLACEHOLDER
    if text.strip() == "":
        return EMPTY_PLACEHOLDER
    return text


def load_tokenizer(model: NomicTextEmbeddingModel) -> Tokenizer:
    """
    Instantiate the huggingface tokenizer.
    :param model: the model doing the embedding
    :return:
    """
    tokdir = Path(__file__).parent / "tokenizers"
    if model == NomicTextEmbeddingModel.nomic_embed_text_v1_5:
        tokenizer = Tokenizer.from_file(str(tokdir / "bert-base-uncased.json"))
        # TODO: change to 8192
        tokenizer.enable_truncation(max_length=2048)
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", pad_to_multiple_of=64)
    else:
        raise Exception(f"Could not determine tokenizer for model: {model.value}")
    return tokenizer


def create_sagemaker_request_for_batch(texts: List[str], tokenizer: Tokenizer):
    """
    Tokenizes and creates a Triton embedding request from a list of texts.

    Args:
        texts: List of texts to be batched.
        tokenizer: Tokenizer to use.

    Returns:
        HTTP Request object for Triton server.
    """
    encoded = tokenizer.encode_batch(texts)

    input_ids = np.array([e.ids for e in encoded], dtype=np.int32)
    attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int32)

    inputs = []
    outputs = []

    inputs.append(aiohttpclient.InferInput("input_ids", list(input_ids.shape), "INT32"))
    inputs.append(
        aiohttpclient.InferInput("attention_mask", list(input_ids.shape), "INT32")
    )

    # Initialize the data
    inputs[0].set_data_from_numpy(input_ids, binary_data=True)
    inputs[1].set_data_from_numpy(attention_mask, binary_data=True)

    # have to set to binary since http doesn't natively support fp16
    outputs.append(aiohttpclient.InferRequestedOutput("embedding", binary_data=True))
    request, header_length = InferenceServerClient.generate_request_body(
        inputs=inputs, outputs=outputs
    )
    return request, header_length


def batch_sagemaker_requests(texts: List[str], batch_size=32):
    """Yield sagemaker triton requests from specified size from list of texts.
    One request will be yielded for each batch.
    Batch size should be set based on GPU provisioned for sagemaker endpoint.

    Args:
        texts: List of texts to be batched.
        batch_size: Size of each batch.

    Yields:
        Batches of data with size equal to batch_size.
    """

    tokenizer = load_tokenizer(NomicTextEmbeddingModel.nomic_embed_text_v1_5)
    for i in range(0, len(texts), batch_size):
        yield create_sagemaker_request_for_batch(texts[i : i + batch_size], tokenizer)


def parse_sagemaker_response(response):
    """
    Parse the response from a sagemaker triton server and return the result as a numpy array.

    Args:
        response: The response from the sagemaker triton server.

    Returns:
        Numpy array of embeddings.
    """
    # Parse json header size length from the response
    header_length_prefix = (
        "application/vnd.sagemaker-triton.binary+json;json-header-size="
    )
    header_length_str = response["ContentType"][len(header_length_prefix) :]

    # Read response body
    result = InferenceServerClient.parse_response_body(
        response["Body"].read(), header_length=int(header_length_str)
    )
    print(result)
    return result.as_numpy("embedding")
