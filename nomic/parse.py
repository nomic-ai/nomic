import base64
import concurrent
import concurrent.futures
import importlib.metadata
import io
import json
import os
import re
import time
import unicodedata
from contextlib import contextmanager
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import requests
from loguru import logger
from pandas import DataFrame
from PIL import Image
from pyarrow import compute as pc
from pyarrow import feather, ipc
from tqdm import tqdm

from .cli import refresh_bearer_token, validate_api_http_response
from .data_inference import (
    NomicDuplicatesOptions,
    NomicEmbedOptions,
    NomicTopicOptions,
    ProjectionOptions,
    convert_pyarrow_schema_for_atlas,
)
from .data_operations import AtlasMapData, AtlasMapDuplicates, AtlasMapEmbeddings, AtlasMapTags, AtlasMapTopics
from .settings import *
from .utils import assert_valid_project_id, download_feather

def _