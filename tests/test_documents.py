import pytest
from pathlib import Path

from nomic.documents import parse, upload_file, extract


def test_parse_document_integration():
    """Integration test for parsing a real PDF file using the actual API."""
    pdf_path = Path(__file__).parent / "Phaedrus.pdf"
    
    # Verify the test file exists
    assert pdf_path.exists(), f"Test PDF file not found at {pdf_path}"
    
    # Parse the document using the real API
    nomic_url = upload_file(pdf_path)
    result = parse(nomic_url)
    
    # Verify we got a result
    assert result is not None
    assert isinstance(result, dict)
    
    print(f"Parsed document result keys: {result.keys()}")
    print(f"Result type: {type(result)}")
    
    # Basic validation that we got some kind of parsed content
    assert len(result) > 0, "Expected non-empty result from document parsing"


def test_extract_document_integration():
    """Integration test for extracting a real PDF file using the actual API."""
    pdf_path = Path(__file__).parent / "Phaedrus.pdf"
    
    # Verify the test file exists
    assert pdf_path.exists(), f"Test PDF file not found at {pdf_path}"
    
    # Parse the document using the real API
    nomic_url = upload_file(pdf_path)
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "speaker": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["speaker", "content"],
        },
    }
   
    result = extract([nomic_url], schema)
    assert result is not None