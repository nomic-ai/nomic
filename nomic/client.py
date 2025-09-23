import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import requests

from nomic.dataset import AtlasClass

client: "AtlasClass | None" = None


def _get_client():
    global client
    if client is None:
        client = AtlasClass()
    return client


@dataclass(frozen=True)
class UploadedFile:
    url: str


class NomicClient:
    """Client for the Nomic Platform API."""

    def upload_file(self, path: "str | os.PathLike[str]") -> UploadedFile:
        """
        Uploads a file to the Nomic Platform.

        Args:
            pdf: The path to the PDF file to upload.

        Returns:
            The response from the Nomic API.
        """
        client = _get_client()

        path = Path(path)

        with path.open("rb") as pdf_file:
            file_type = path.suffix.lower()
            if file_type == ".pdf":
                content_type = "application/pdf"
            # elif file_type == ".docx":
            #   content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            # elif file_type == ".doc":
            #   content_type = "application/msword"
            # elif file_type == ".txt":
            #   content_type = "text/plain"
            # elif file_type == ".pptx":
            #   content_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            # elif file_type == ".ppt":
            #   content_type = "application/vnd.ms-powerpoint"
            # elif file_type == ".csv":
            #   content_type = "text/csv"
            # elif file_type == ".xlsx":
            #   content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            # elif file_type == ".xls":
            #   content_type = "application/vnd.ms-excel"
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            response = client._post(
                "/v1/upload",
                json=dict(files=[{"id": path.name, "size": path.stat().st_size, "content_type": content_type}]),
            )
            response.raise_for_status()

            values = response.json()

            # Extract from the files array
            file_info = values["files"][0]
            upload_url = file_info["upload_url"]
            nomic_url = file_info["nomic_url"]

            # upload the file to the designated pre-signed url
            resp = requests.put(upload_url, data=pdf_file, headers={"x-amz-server-side-encryption": "AES256"})

        resp.raise_for_status()
        return UploadedFile(url=nomic_url)

    @staticmethod
    def _wait_for_task_completion(task_id: str) -> str:
        """
        Takes a nomic-formatted url, waits for it to be parsed,
        and returns the url of the parsed document.

        Args:
            task_id: The ID of the task to wait for.

        Returns:
            The url of the parsed document.
        """
        client = _get_client()
        while True:
            response = client._get(f"/v1/status/{task_id}")
            response.raise_for_status()
            value = response.json()
            if value["status"] == "COMPLETED":
                return value["result_url"]
            if value["status"] == "FAILED":
                raise Exception(f"Task failed: {value['error']}")
            time.sleep(1)

    def parse(self, file: UploadedFile) -> "dict[str, Any]":
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

        response = client._post("/v1/parse", json={"file_url": file.url})
        response.raise_for_status()
        resp = response.json()
        task_id = resp["task_id"]

        completed_url = self._wait_for_task_completion(task_id)
        completed_response = requests.get(completed_url)
        completed_response.raise_for_status()
        return completed_response.json()

    def extract(self, files: "list[UploadedFile]", schema: "dict[str, Any]") -> Any:
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

        response = client._post(
            "/v1/extract", json={"file_urls": [file.url for file in files], "extraction_schema": schema}
        )
        resp = response.json()
        task_id = resp["task_id"]

        completed_url = self._wait_for_task_completion(task_id)
        completed_response = requests.get(completed_url)
        completed_response.raise_for_status()
        return completed_response.json()
