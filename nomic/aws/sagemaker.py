import concurrent.futures
import io
import json
import logging
import multiprocessing as mp
from pathlib import PosixPath
from typing import List, Optional, Tuple, Union

import boto3
import PIL
import PIL.Image
import sagemaker
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    return resp["embeddings"]


def batch_transform_text(
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


def embed_text(
    texts: List[str],
    sagemaker_endpoint: str,
    region_name: str,
    task_type: str = "search_document",
    batch_size: int = 32,
    dimensionality: int = 768,
    binary: bool = False,
):
    """
    Embed a list of texts using a sagemaker model endpoint.

    Args:
        texts: List of texts to be embedded.
        sagemaker_endpoint: The sagemaker endpoint to use.
        region_name: AWS region sagemaker endpoint is in.
        task_type: The task type to use when embedding.
        batch_size: Size of each batch. Default is 32.
        dimensionality: Number of dimensions to return. Options are (64, 128, 256, 512, 768).
        binary: Whether to return binary embeddings.

    Returns:
        Dictionary with "embeddings" (python 2d list of floats), "model" (sagemaker endpoint used to generate embeddings).
    """

    if len(texts) == 0:
        logger.warning("No texts to embed.")
        return None

    assert task_type in [
        "search_query",
        "search_document",
        "classification",
        "clustering",
    ], f"Invalid task type: {task_type}"

    assert dimensionality in (
        64,
        128,
        256,
        512,
        768,
    ), f"Invalid number of dimensions: {dimensionality}"

    client = boto3.client("sagemaker-runtime", region_name=region_name)
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = json.dumps(
            {
                "texts": texts[i : i + batch_size],
                "binary": binary,
                "dimensionality": dimensionality,
                "task_type": task_type,
            }
        )
        response = client.invoke_endpoint(EndpointName=sagemaker_endpoint, Body=batch, ContentType="application/json")
        embeddings.extend(parse_sagemaker_response(response))

    return {
        "embeddings": embeddings,
        "model": "nomic-embed-text-v1.5",
        "usage": {},
    }


# only way I could get sagemaker with multipart to work
def prepare_multipart_request(images: List[Tuple[str, bytes]]) -> Tuple[bytes, bytes]:
    # Prepare the multipart body
    boundary = b"---------------------------Boundary"
    body = b""
    for i, (name, img_bytes) in enumerate(images):
        body += b"--" + boundary + b"\r\n"
        body += f'Content-Disposition: form-data; name="{name}"; filename="image_{i}.jpg"\r\n'.encode("utf-8")
        body += b"Content-Type: image/jpeg\r\n\r\n"
        body += img_bytes + b"\r\n"
    body += b"--" + boundary + b"--\r\n"

    return body, boundary


def preprocess_image(images: List[Union[str, "PIL.Image.Image", bytes]]) -> Tuple[bytes, bytes]:
    """
    Preprocess a list of images for embedding using a sagemaker model.

    Args:
        images: List of images to be embedded.

    Returns:
        List of images formatted for sagemaker embedding.
    """
    encoded_images = []
    for image in images:
        if isinstance(image, str):
            image = PIL.Image.open(image)
        elif isinstance(image, bytes):
            image = PIL.Image.open(io.BytesIO(image))
        elif isinstance(image, PosixPath):
            image = PIL.Image.open(image)
        else:
            assert isinstance(image, PIL.Image.Image), f"Invalid image type: {type(image)}"
        image = image.convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        encoded_images.append(("image_data", buffered.getvalue()))

    body, boundary = prepare_multipart_request(encoded_images)
    return body, boundary


def sagemaker_image_request(
    images: List[Union[str, bytes, "PIL.Image.Image"]], sagemaker_endpoint: str, region_name: str
):
    body, boundary = preprocess_image(images)

    client = boto3.client("sagemaker-runtime", region_name=region_name)
    response = client.invoke_endpoint(
        EndpointName=sagemaker_endpoint,
        Body=body,
        ContentType=f'multipart/form-data; boundary={boundary.decode("utf-8")}',
    )

    return parse_sagemaker_response(response)


def embed_image(
    images: List[Union[str, "PIL.Image.Image", bytes]],
    sagemaker_endpoint: str,
    region_name: str,
    model_name="nomic-embed-vision-v1.5",
    batch_size=16,
) -> dict:
    embeddings = []

    pbar = tqdm(total=len(images))
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        embeddings.extend(
            sagemaker_image_request(batch, sagemaker_endpoint=sagemaker_endpoint, region_name=region_name)
        )
        pbar.update(len(batch))

    return {
        "embeddings": embeddings,
        "model": model_name,
        "usage": {},
    }


def batch_transform_image(
    s3_input_path: str,
    s3_output_path: str,
    region_name: str,
    arn: Optional[str] = None,
    role: Optional[str] = None,
    max_payload: Optional[int] = 6,
    instance_type: str = "ml.g4dn.xlarge",
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
        assemble_with="Line",
        max_payload=max_payload,
    )
    embedder.transform(
        data=s3_input_path,
        content_type="application/x-image",
        split_type=None,
        wait=wait,
        logs=logs,
    )
    job_name = None
    if embedder.latest_transform_job is not None:
        job_name = embedder.latest_transform_job.name
    return job_name
