import importlib.metadata
from pathlib import Path
import time
from typing import Any, Optional, Union
import jsonschema
import requests

from nomic.dataset import AtlasClass

client: Optional[AtlasClass] = None

try:
    version = importlib.metadata.version("nomic")
except Exception:
    version = "unknown"

def _get_client():
  global client
  if client is None:
    client = AtlasClass()
  return client

def upload_file(pdf: Union[str, Path]) -> str:
  """
  Uploads a file to the Nomic Platform.

  Args:
    pdf: The path to the PDF file to upload.

  Returns:
    The response from the Nomic API.
  """
  client = _get_client()  

  pdf = Path(pdf)
  if not pdf.exists():
    raise FileNotFoundError(f"PDF file not found: {pdf}")
  if not pdf.is_file():
    raise NotADirectoryError(f"PDF file is not a file: {pdf}")
  if not pdf.suffix == ".pdf":
    raise ValueError(f"PDF file must have a .pdf extension: {pdf}")

  file_type = pdf.suffix.lower()
  if file_type == ".pdf":
    file_type = "application/pdf"
  # elif file_type == ".docx":
  #   file_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
  # elif file_type == ".doc":
  #   file_type = "application/msword"
  # elif file_type == ".txt":
  #   file_type = "text/plain"
  # elif file_type == ".pptx":
  #   file_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
  # elif file_type == ".ppt":
  #   file_type = "application/vnd.ms-powerpoint"
  # elif file_type == ".csv":
  #   file_type = "text/csv"
  # elif file_type == ".xlsx":
  #   file_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
  # elif file_type == ".xls":
  #   file_type = "application/vnd.ms-excel"
  else:
    raise ValueError(f"Unsupported file type: {file_type}")
  
  response = client.post("/v1/upload", json=
    dict(files=[{
      "id": str(pdf),
     "size": pdf.stat().st_size,
     "content_type": "application/pdf"
  }]))
  
  values = response.json()
  
  # Extract from the files array
  file_info = values["files"][0]
  upload_url = file_info["upload_url"]
  nomic_url = file_info["nomic_url"]
  
  # upload the file to the designated pre-signed url
  resp = requests.put(upload_url, data=pdf.open('rb').read(), headers={'x-amz-server-side-encryption': 'AES256'})
  resp.raise_for_status()
  return nomic_url



def _wait_for_task_completion(task_id: str) -> str:
  """
  Takes a nomic-formatted url, waits for it to be parsed, 
  and returns the url of the parsed document.

  Args:
    nomic_url: The URL of the document to parse.

  Returns:
    The url of the parsed document.
  """
  client = _get_client()
  while True:
    response = client.get(f"/v1/status/{task_id}")
    value = response.json()
    if value["status"] == "COMPLETED":
      return value["result_url"]
    elif value["status"] == "FAILED":
      raise Exception(f"Task failed: {value['error']}")
    else:
      time.sleep(1)

def parse(nomic_url: str):
  """
  Parses a document into a structured JSON representation.

  Example:
    Complete end-to-end workflow with upload and parsing:

    ```python
    from nomic.documents import upload_file, parse

    # Upload a PDF file
    nomic_url = upload_file("my_document.pdf")

    # Parse the document
    result = parse(nomic_url)
    print(result)
    ```
  """

  client = _get_client()

  response = client.post("/v1/parse", json={"file_url": nomic_url})
  resp = response.json()
  task_id = resp["task_id"]

  completed_url = _wait_for_task_completion(task_id)
  completed_response = requests.get(completed_url)
  completed_response.raise_for_status()
  return completed_response.json()

def extract(file_urls: list[str], schema: dict[str, Any]):
  """
  Extracts structured data from documents.

  Args:
    file_urls: List of Nomic URLs of documents to extract from.
    schema: A JSON schema defining the structure of data to extract.
  
  Returns:
    The extracted data matching the provided schema.

  Example:
    Complete end-to-end workflow with upload and extraction:

    ```python
    from nomic.documents import upload_file, extract

    # Upload a PDF file
    pdf_path = Path()
    nomic_url = upload_file("my_document.pdf")

    # Define extraction schema
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "speaker": {"type": "string"},
                "content": {"type": "string"},
            }
        },
    }

    # Extract structured data
    result = extract([nomic_url], schema)
    print(result)
    ```
  """
  jsonschema.Draft7Validator.check_schema(schema)

  client = _get_client()

  response = client.post("/v1/extract", json={"file_urls": file_urls, "extraction_schema": schema})
  resp = response.json()
  task_id = resp["task_id"]

  completed_url = _wait_for_task_completion(task_id)
  completed_response = requests.get(completed_url)
  completed_response.raise_for_status()
  return completed_response.json()
