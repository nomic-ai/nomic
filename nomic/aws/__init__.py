def _check_aws_dependencies():
    try:
        import boto3
    except ImportError as e:
        missing_package = str(e).split("No module named ")[-1].strip("'")
        raise ImportError(
            f"The '{missing_package}' package is required for this feature. "
            "Please install it by running 'pip install nomic[aws]'."
        )


_check_aws_dependencies()

from .sagemaker import embed_image, embed_text
