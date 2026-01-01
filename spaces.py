from typing import Any

import boto3

SPACES_ENDPOINT_URL = "https://sfo3.digitaloceanspaces.com"
SPACES_BUCKET_NAME = "alpha-production-bucket"
SPACES_ACCESS_KEY_ID = "DO801RYGDGV7U8T7EVYD"
SPACES_SECRET_ACCESS_KEY = "JCnthU/AJBYFF/nb8VjdmY43QcBsQyRS4VXKimDiqKI"


def get_s3_client() -> Any:
    session = boto3.session.Session()
    client = session.client(
        service_name="s3",
        region_name="nyc3",
        endpoint_url=SPACES_ENDPOINT_URL,
        aws_access_key_id=SPACES_ACCESS_KEY_ID,
        aws_secret_access_key=SPACES_SECRET_ACCESS_KEY,
    )

    return client


def stream_object_from_url(object_url: str) -> bytes:
    print(f"Streaming object from {object_url}")

    s3_client = get_s3_client()
    s3_object = s3_client.get_object(Bucket=SPACES_BUCKET_NAME, Key=object_url)
    object_content = s3_object["Body"].read()

    print(f"Object content {len(object_content)} bytes")

    return object_content


def save_object(object_bytes: bytes, object_name: str, directory: str) -> str:
    object_url = f"{directory}/{object_name}"

    print(f"Saving object to {object_url}")

    s3_client = get_s3_client()
    s3_client.put_object(
        Bucket=SPACES_BUCKET_NAME, Key=object_url, Body=object_bytes
    )

    return object_url
