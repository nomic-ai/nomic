import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar, overload
from urllib.parse import urlparse

import jsonschema
import requests

from nomic.dataset import AtlasClass

from .client_models import (
    ChunkerType,
    ChunkOptions,
    ContentExtractionMode,
    ExtractOptions,
    ExtractRequest,
    FigureSummaryOptions,
    OcrLanguage,
    ParseOptions,
    ParseRequest,
    TableSummaryOptions,
)

__all__ = [
    "NomicClient",
    "PlatformTask",
    "TaskFailed",
    "TaskPending",
    "UploadedFile",
    # Client models
    "ChunkOptions",
    "ChunkerType",
    "ContentExtractionMode",
    "ExtractOptions",
    "FigureSummaryOptions",
    "OcrLanguage",
    "ParseOptions",
    "TableSummaryOptions",
]

MAX_FAILRESP_LENGTH = 1_000  # chars

T = TypeVar("T")

client: "AtlasClass | None" = None


def get_client():
    global client
    if client is None:
        client = AtlasClass()
    return client


@dataclass(frozen=True)
class UploadedFile:
    url: str


class Sentinel(Enum):
    Nothing = auto()


class TaskPending(Exception):
    pass


class TaskFailed(Exception):
    pass


class PlatformTask(Generic[T]):
    """
    An object representing a task on the Nomic Platform.

    Attributes:
        id: The ID of the task.
    """

    _id: str
    _result: "T | Sentinel"

    def __init__(self, id: str):
        self._id = id
        self._result = Sentinel.Nothing

    @property
    def id(self) -> str:
        return self._id

    def get(self, timeout: "float | None" = None, *, block: bool = True) -> T:
        """
        Waits for the task to complete and returns the result.

        Args:
            timeout: The maximum time to wait for the task to complete.
            block: Whether to block until the task is complete.

        Returns:
            The result of the task.

        Raises:
            TaskPending: If the task is not complete and block is True.
            TaskFailed: If the task fails.
        """
        if self._result is not Sentinel.Nothing:
            return self._result
        client = get_client()
        start_time = time.time()
        while True:
            response = client._get(f"/v1/status/{self._id}")
            raise_for_status_with_body(response)
            status_resp = response.json()
            if status_resp["status"] == "COMPLETED":
                break
            if status_resp["status"] == "FAILED":
                raise TaskFailed(status_resp["error"])
            if not block:
                raise TaskPending
            sleeptime = 1  # poll interval
            if timeout is not None:
                end_time = start_time + timeout
                if end_time < (now := time.time()):
                    raise TaskPending
                sleeptime = min(sleeptime, end_time - now)
            time.sleep(sleeptime)

        completed_response = requests.get(status_resp["result_url"])
        raise_for_status_with_body(completed_response)

        result = status_resp.pop("result", {})
        result.pop("result_url", None)
        result.pop("result", None)
        result["result"] = completed_response.json()
        result["result"].pop("status", None)
        result["result"].pop("error", None)
        self._result = result
        return result


class NomicClient:
    """Client for the Nomic Platform API."""

    def upload_file(self, path: "str | os.PathLike[str]") -> UploadedFile:
        """
        Uploads a file to the Nomic Platform.

        Args:
            path: The path to the PDF file to upload.

        Returns:
            An UploadedFile object representing the uploaded file.
        """
        client = get_client()

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
            raise_for_status_with_body(response)

            values = response.json()

            # Extract from the files array
            file_info = values["files"][0]
            upload_url = file_info["upload_url"]
            nomic_url = file_info["nomic_url"]

            # upload the file to the designated pre-signed url
            resp = requests.put(upload_url, data=pdf_file, headers={"x-amz-server-side-encryption": "AES256"})

        raise_for_status_with_body(resp)
        return UploadedFile(url=nomic_url)

    @overload
    def parse(
        self, file: "str | UploadedFile", *, options: "ParseOptions | None" = ..., block: Literal[True] = ...
    ) -> "dict[str, Any]": ...
    @overload
    def parse(
        self, file: "str | UploadedFile", *, options: "ParseOptions | None" = ..., block: Literal[False]
    ) -> PlatformTask["dict[str, Any]"]: ...
    @overload
    def parse(self, file: "str | UploadedFile", *, options: "ParseOptions | None" = ..., block: bool) -> Any: ...

    def parse(self, file: "str | UploadedFile", *, options: "ParseOptions | None" = None, block: bool = True) -> Any:
        """
        Parses a document into a structured JSON representation.

        Args:
            file: The file to parse. Can be a string URL or an UploadedFile object.
            options: Optional ParseOptions to customize parsing behavior (OCR settings, chunking, etc.).
            block: Whether to block until the task is complete.

        Returns:
            By default, returns the parsed document. If block is False, returns a PlatformTask that can be used to get
            the result.

        Raises:
            ValueError: If an invalid URL is passed.
            TaskFailed: If block is True and the task fails.

        Example:
            Complete end-to-end workflow with upload and parsing:

            ```python
            from nomic.client import NomicClient

            # Upload a PDF file
            client = NomicClient()
            file = client.upload_file("my_document.pdf")

            # Parse the document
            result = client.parse(file)
            print(result)
            ```
        """
        client = get_client()

        file_url = self._file_to_url(file)
        if options is not None:
            request = ParseRequest(file_url=file_url, options=options)
        else:
            request = ParseRequest(file_url=file_url)
        payload = request.model_dump(mode="json", exclude_unset=True)

        response = client._post("/v1/parse", json=payload)
        raise_for_status_with_body(response)
        task = PlatformTask(response.json()["task_id"])
        if block:
            return task.get()
        return task

    @overload
    def extract(
        self,
        files: "str | UploadedFile | Sequence[str | UploadedFile]",
        schema: "dict[str, Any]",
        *,
        options: "ExtractOptions | None" = ...,
        block: Literal[True] = ...,
    ) -> Any: ...
    @overload
    def extract(
        self,
        files: "str | UploadedFile | Sequence[str | UploadedFile]",
        schema: "dict[str, Any]",
        *,
        options: "ExtractOptions | None" = ...,
        block: Literal[False],
    ) -> PlatformTask[Any]: ...
    @overload
    def extract(
        self,
        files: "str | UploadedFile | Sequence[str | UploadedFile]",
        schema: "dict[str, Any]",
        *,
        options: "ExtractOptions | None" = ...,
        block: bool,
    ) -> Any: ...

    def extract(
        self,
        files: "str | UploadedFile | Sequence[str | UploadedFile]",
        schema: "dict[str, Any]",
        *,
        options: "ExtractOptions | None" = None,
        block: bool = True,
    ) -> Any:
        """
        Extracts structured data from documents.

        Args:
            files: List of uploaded files to extract from.
            schema: A JSON schema defining the structure of data to extract.
            options: Optional ExtractOptions to customize extraction behavior (system prompt, etc.).
            block: Whether to block until the task is complete.

        Returns:
            By default, returns the extracted data matching the provided schema. If block is False, returns a PlatformTask
            that can be used to get the result.

        Raises:
            ValueError: If an invalid URL is passed.
            TaskFailed: If block is True and the task fails.

        Example:
            Complete end-to-end workflow with upload and extraction:

            ```python
            from nomic.client import NomicClient

            # Upload a PDF file
            client = NomicClient()
            file = client.upload_file("my_document.pdf")

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
            result = client.extract(file, schema)
            print(result)
            ```
        """
        jsonschema.Draft7Validator.check_schema(schema)

        if isinstance(files, (str, UploadedFile)):
            files = [files]

        client = get_client()

        request = ExtractRequest(
            file_urls=list(map(self._file_to_url, files)),
            extraction_schema=schema,
            system_prompt=options.system_prompt if options is not None else None,
        )
        payload = request.model_dump(mode="json", exclude_unset=True)

        response = client._post("/v1/extract", json=payload)
        raise_for_status_with_body(response)
        task = PlatformTask(response.json()["task_id"])
        if block:
            return task.get()
        return task

    @staticmethod
    def _file_to_url(file: "str | UploadedFile") -> str:
        if isinstance(file, UploadedFile):
            return file.url
        parsed = urlparse(file)
        if parsed.scheme in ("nomic", "http", "https"):
            return file
        if parsed.scheme == "file" or (not parsed.scheme and Path(file).exists()):
            raise ValueError(
                f"Cannot directly pass local file to platform: {file!r}\nPlease use upload_file() to upload it first."
            )
        if not parsed.scheme:
            raise ValueError(f"Invalid URL: {file!r}")
        raise ValueError(f"Unsupported scheme {parsed.scheme!r} for URL {file!r}")


def raise_for_status_with_body(resp: requests.Response) -> None:
    """
    Raises HTTPError if the response is not successful.

    Like Response.raise_for_status, but includes the (truncated) response body in the
    exception message for improved diagnostics.

    Args:
        resp: The response to check

    Raises:
        requests.HTTPError: If the response is not successful
    """
    http_error_msg = ""
    if isinstance(resp.reason, bytes):
        # We attempt to decode utf-8 first because some servers
        # choose to localize their reason strings. If the string
        # isn't utf-8, we fall back to iso-8859-1 for all other
        # encodings. (See PR #3538)
        try:
            reason = resp.reason.decode("utf-8")
        except UnicodeDecodeError:
            reason = resp.reason.decode("iso-8859-1")
    else:
        reason = resp.reason

    if 400 <= resp.status_code < 500:
        http_error_msg = f"{resp.status_code} Client Error: {reason} for url: {resp.url}"

    elif 500 <= resp.status_code < 600:
        http_error_msg = f"{resp.status_code} Server Error: {reason} for url: {resp.url}"

    if http_error_msg:
        if (ctype := resp.headers.get("content-type")) is not None:
            http_error_msg += f"\nContent-Type: {ctype}"
        http_error_msg += f"\nBody: {format_body(resp)}"
        raise requests.HTTPError(http_error_msg, response=resp)


def format_body(resp: requests.Response) -> str:
    text = None
    if (ctype := resp.headers.get("content-type")) and "application/json" in ctype.lower():
        try:
            data = resp.json()
            text = json.dumps(data, indent=2, ensure_ascii=False)
        except Exception:
            pass
    if text is None:
        text = resp.text

    limit = MAX_FAILRESP_LENGTH
    if len(text) > limit:
        return text[:limit] + f"\n… [truncated {len(text) - limit} chars]"
    return text
