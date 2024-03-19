# Note: This file contains methods to prepare embedding requests for Sagemaker.
# It may make sense to move code eventually to embed.py or somewhere more generic but
# it currently lives here to separate out dependencies.

import hashlib
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


import numpy as np
import tritonclient.http.aio as aiohttpclient
from tritonclient.http._utils import _get_inference_request
from tokenizers import Tokenizer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


text_embedding_model_info = {
    'nomic-embed-text-v1.5': {
        'dim': 768,
        'max_length': 2048,
        'pad_id': 0,
        'recommended_dims': [768, 512, 384, 256, 128],
    },
}

NULL_PLACEHOLDER = hashlib.md5(b'nomic null').hexdigest()
EMPTY_PLACEHOLDER = hashlib.md5(b'nomic empty').hexdigest()


class NomicTextEmbeddingModel(Enum):
    nomic_embed_text_v1_5 = 'nomic-embed-text-v1.5'

    def hamming_capable(self):
        return self == NomicTextEmbeddingModel.nomic_embed_text_v1_5

    def matryoshka_capable(self):
        return self == NomicTextEmbeddingModel.nomic_embed_text_v1_5

    def dim(self):
        return text_embedding_model_info[self.value]['dim']

    def recommended_dims(self):
        return text_embedding_model_info[self.value].get('recommended_dims') or [self.dim()]

    def max_length(self):
        return text_embedding_model_info[self.value]['max_length']

    def pad_token(self) -> int:
        return text_embedding_model_info[self.value]['pad_id']


def null_empty_placeholder(text: Optional[str]) -> str:
    """
    If text is null or empty replace with a placeholder nonsense string.
    """
    if text is None:
        return NULL_PLACEHOLDER
    if text.strip() == '':
        return EMPTY_PLACEHOLDER
    return text


def load_tokenizer(model: NomicTextEmbeddingModel) -> Tokenizer:
    """
    Instantiate the huggingface tokenizer.
    :param model: the model doing the embedding
    :return:
    """
    tokdir = Path(__file__).parent / 'tokenizers'
    if model == NomicTextEmbeddingModel.nomic_embed_text_v1_5:
        tokenizer = Tokenizer.from_file(str(tokdir / 'bert-base-uncased.json'))
        tokenizer.enable_truncation(max_length=2048)
        tokenizer.enable_padding(pad_id=0, pad_token='[PAD]', pad_to_multiple_of=64)
    else:
        raise Exception(f"Could not determine tokenizer for model: {model.value}")
    return tokenizer

@contextmanager
def no_pad(tok):
    if tok.padding is not None:
        keep = tok.padding
        tok.no_padding()
        yield tok
        tok.enable_padding(**keep)
    else:
        yield tok

@contextmanager
def no_trunc(tok):
    if tok.truncation is not None:
        keep = tok.truncation
        tok.no_truncation()
        yield tok
        tok.enable_truncation(**keep)
    else:
        yield tok

def tokenize_text(text,
                  tokenizer = None, 
                  model: Optional[NomicTextEmbeddingModel] = None, 
                  add_special_tokens=False):
    if tokenizer is None:
        if model is None:
            raise ValueError("Either tokenizer or model must be provided")
        tokenizer = load_tokenizer(model)
    # padding and truncation are handled by tokenizer
    all_tokens = tokenizer.encode(text, add_special_tokens=add_special_tokens).ids
    if len(all_tokens) == 0:
        logger.warning(f'Zero tokens generated from text.')
        all_tokens = tokenize_text(EMPTY_PLACEHOLDER, add_special_tokens=False).ids
    

def create_sm_request_for_batch(texts):
    all_tokens = []
    for text in texts:
        all_tokens.append(tokenize_text(text))

    input_ids = np.array(all_tokens, dtype=np.int32)
    mlen = max(len(tokens) for tokens in all_tokens)
    attention_mask = [([1] * len(tokens)) + [0] * (mlen - len(tokens)) for tokens in all_tokens]
    attention_mask = np.array(attention_mask, dtype=np.int32)

    inputs = []
    outputs = []

    inputs.append(aiohttpclient.InferInput("input_ids", input_ids.shape, "INT32"))
    inputs.append(aiohttpclient.InferInput("attention_mask", input_ids.shape, "INT32"))

    # Initialize the data
    inputs[0].set_data_from_numpy(input_ids, binary_data=True)
    inputs[1].set_data_from_numpy(attention_mask, binary_data=True)

    # have to set to binary since http doesn't natively support fp16
    outputs.append(aiohttpclient.InferRequestedOutput("embedding", binary_data=True))

    request = _get_inference_request(inputs=inputs, 
                                     request_id="", 
                                     outputs=outputs, 
                                     sequence_id="", 
                                     sequence_start=0, 
                                     sequence_end=None, 
                                     priority=None, 
                                     timeout=None, 
                                     custom_parameters=None)
    return request