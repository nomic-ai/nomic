# Note: This file contains methods to prepare embedding requests for Sagemaker.
# It may make sense to move code eventually to embed.py or somewhere more generic but
# it currently lives here to separate out dependencies.

import hashlib
import json
import logging
from enum import Enum
from typing import List

import boto3
import numpy as np
from tqdm import tqdm

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


def parse_sagemaker_response(response):
    """
    Parse response from a sagemaker triton server and return the embedding as a numpy array.

    Args:
        response: The response from the sagemaker triton server.

    Returns:
        np.float16 array of embeddings.
    """
    # Parse json header size length from the response
    resp = json.loads(response["Body"].read().decode())
    return np.array(resp["embeddings"], dtype=np.float16)


def embed_texts(
    texts: List[str], sagemaker_endpoint: str, region_name: str, batch_size=32
):
    """
    Embed a list of texts using a sagemaker model endpoint.

    Args:
        texts: List of texts to be embedded.
        sagemaker_endpoint: The sagemaker endpoint to use.
        region_name: AWS region sagemaker endpoint is in.
        batch_size: Size of each batch.

    Returns:
        np.float16 array of embeddings.
    """
    if len(texts) == 0:
        logger.warning("No texts to embed.")
        return None

    client = boto3.client("sagemaker-runtime", region_name=region_name)
    embeddings = []

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = json.dumps({"texts": texts[i : i + batch_size]})
        response = client.invoke_endpoint(
            EndpointName=sagemaker_endpoint, Body=batch, ContentType="application/json"
        )
        embeddings.append(parse_sagemaker_response(response))
    return np.vstack(embeddings)
