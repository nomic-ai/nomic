# Note: This file contains methods to prepare embedding requests for Sagemaker.
# It may make sense to move code eventually to embed.py or somewhere more generic but
# it currently lives here to separate out dependencies.

import asyncio
import hashlib
import logging
from collections.abc import Iterable
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, AsyncIterator, Callable, Dict, Generic, List, Optional, Tuple, TypeVar
from weakref import WeakKeyDictionary


import numpy as np
import psutil
import pyarrow as pa
import tritonclient.http.aio as aiohttpclient
from tokenizers import Tokenizer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


NOMIC_EMBED_TEXT_V1_5 = {
    'dim': 768,
    'max_length': 2048,
    'pad_id': 0,
    'recommended_dims': [768, 512, 384, 256, 128],
}

NULL_PLACEHOLDER = hashlib.md5(b'nomic null').hexdigest()
EMPTY_PLACEHOLDER = hashlib.md5(b'nomic empty').hexdigest()






