import json
import logging
from typing import List, Optional

import boto3
import numpy as np
import sagemaker
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# NOTE: Currently Sagemaker only supports nomic-embed-text-v1.5 model.

SAGEMAKER_MODELS = {"nomic-embed-text-v1.5": {"us-east-2": "TODO: ARN"}}


def _get_supported_regions(model: str):
    return SAGEMAKER_MODELS[model].keys()


def _get_model_and_region_for_arn(arn: str):
    for model in SAGEMAKER_MODELS:
        for region, arn in SAGEMAKER_MODELS[model].items():
            if arn == arn:
                return model, region
    raise ValueError(f"Model package arn {arn} not supported.")


def _get_sagemaker_role():
    try:
        return sagemaker.get_execution_role()
    except ValueError:
        raise ValueError("Unable to fetch sagemaker execution role. Please provide a role.")


def parse_sagemaker_response(response):
    """
    Parse response from a sagemaker server and return the embedding as a numpy array.

    Args:
        response: The response from the sagemaker server.

    Returns:
        np.float16 array of embeddings.
    """
    # Parse json header size length from the response
    resp = json.loads(response["Body"].read().decode())
    return np.array(resp["embeddings"], dtype=np.float16)


def batch_transform(
    s3_input_path: str,
    s3_output_path: str,
    region_name: str,
    arn: Optional[str] = None,
    role: Optional[str] = None,
    max_payload: Optional[int] = 6,
    instance_type: str = "ml.p3.2xlarge",
    n_instances: int = 1,
    wait: bool = True,
    logs: bool = True,
):
    """
    Batch transform a list of texts using a sagemaker model.

    Args:
        s3_input_path: S3 path to the input data. Input data should be a csv file without any column headers
            with each line containing a single text.
        s3_output_path: S3 path to save the output embeddings. Embeddings will be in order of the input data.
        region_name: AWS region to use.
        arn: The model package arn to use.
        role: Arn of the IAM role to use (must have permissions to S3 data as well).
        max_payload: The maximum payload size in megabytes.
        instance_type: The instance type to use.
        n_instances: The number of instances to use.
        wait: Whether method should wait for job to finish.
        logs: Whether to log the job status (only meaningful if wait is True).
    Returns:
        The job name.
    """
    if arn is None:
        raise ValueError("model package arn is currently required.")
        if region_name is None or model_name is None:
            raise ValueError("region_name and model_name is required if arn is not provided.")
        if region_name not in _get_supported_regions(model_name):
            raise ValueError(f"Model {model_name} not supported in region {region_name}.")
        arn = SAGEMAKER_MODELS[model_name][region_name]

    if role is None:
        logger.info("No role provided. Using default sagemaker role.")
        role = _get_sagemaker_role()

    sm_client = boto3.client("sagemaker", region_name=region_name)
    sm_session = sagemaker.Session(
        boto_session=boto3.Session(region_name=region_name),
        sagemaker_client=sm_client,
    )
    model = sagemaker.ModelPackage(
        name=arn.split("/")[-1],
        role=role,
        model_data=None,
        sagemaker_session=sm_session,  # makes sure the right region is used
        model_package_arn=arn,
    )
    embedder = model.transformer(
        instance_count=n_instances,
        instance_type=instance_type,
        output_path=s3_output_path,
        strategy="MultiRecord",
        assemble_with="Line",
        max_payload=max_payload,
    )
    embedder.transform(
        data=s3_input_path,
        content_type="text/csv",
        split_type="Line",
        wait=wait,
        logs=logs,
    )
    job_name = None
    if embedder.latest_transform_job is not None:
        job_name = embedder.latest_transform_job.name
    return job_name


def embed_texts(texts: List[str], sagemaker_endpoint: str, region_name: str, batch_size=32):
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

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = json.dumps({"texts": texts[i : i + batch_size]})
        response = client.invoke_endpoint(EndpointName=sagemaker_endpoint, Body=batch, ContentType="application/json")
        embeddings.append(parse_sagemaker_response(response))
    return np.vstack(embeddings)
