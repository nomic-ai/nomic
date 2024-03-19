from nomic.aws.sagemaker import batch_sagemaker_requests
import pytest

def test_batch_sagemaker_requests():
    texts = [
        "This is a test", 
        "This is another test",
        "This is a third test",
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumps over the lazy dog",
    ]
    
    num_request = 0
    for request in batch_sagemaker_requests(texts, batch_size=2):
        num_request += 1
        assert request is not None
    assert num_request == 3
