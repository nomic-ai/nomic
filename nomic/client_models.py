"""Client-side models for the Nomic Platform API."""

from enum import Enum

from pydantic import BaseModel, Field

__all__ = [
    "ContentExtractionMode",
    "OcrLanguage",
    "TableSummaryOptions",
    "FigureSummaryOptions",
    "ParseOptions",
    "ExtractOptions",
    "ParseRequest",
    "ExtractRequest",
]


class ContentExtractionMode(str, Enum):
    """The overall strategy for extracting content from the document."""

    Metadata = "metadata"  # Disable all OCR. Only use embedded document text.
    Hybrid = "hybrid"  # Use a VLM for tables, and run an OCR model on all bitmaps found in the document.
    Ocr = "ocr"  # Use a VLM for tables. Run an OCR model on full pages.


class OcrLanguage(str, Enum):
    """Language selection for OCR."""

    English = "en"
    Latin = "latin"
    Chinese_Japanese_English = "zh_ja_en"


class TableSummaryOptions(BaseModel):
    """Options for generating table summaries."""

    enabled: bool = Field(
        default=False,
        description="Whether to generate a summary of table content",
    )


class FigureSummaryOptions(BaseModel):
    """Options for generating figure summaries."""

    enabled: bool = Field(
        default=True,
        description="Whether to generate a summary of figure content",
    )


class ParseOptions(BaseModel):
    """Options to customize document parsing."""

    content_extraction_mode: ContentExtractionMode = Field(
        default=ContentExtractionMode.Hybrid,
        description="The overall strategy for extracting content from the document",
    )
    ocr_language: OcrLanguage = Field(
        default=OcrLanguage.English,
        description="Language selection for OCR",
    )
    table_summary: TableSummaryOptions | None = Field(
        default=None,
        description="Options for generating table summaries",
    )
    figure_summary: FigureSummaryOptions | None = Field(
        default=None,
        description="Options for generating figure summaries",
    )


class ExtractOptions(BaseModel):
    """Options to customize document extraction."""

    system_prompt: str | None = Field(
        default=None,
        description="Custom system prompt to guide the AI extraction process across the entire file. "
        "Use this to provide specific instructions, context, or constraints for how information "
        "should be extracted and formatted according to your requirements.",
    )


class ParseRequest(BaseModel):
    """Request model for parsing a document."""

    file_url: str = Field(description="File URL to process")
    options: ParseOptions = Field(
        default_factory=ParseOptions,
        description="Options to customize document parsing",
    )


class ExtractRequest(BaseModel):
    """Request model for extracting data from documents."""

    file_urls: list[str] = Field(description="List of file URLs to process for extraction")
    extraction_schema: dict = Field(description="JSON schema defining the structure of data to extract")
    system_prompt: str | None = Field(
        default=None,
        description="Custom system prompt to guide the AI extraction process",
    )
