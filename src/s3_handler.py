import boto3
from pathlib import  Path
from tqdm import tqdm
from botocore.exceptions import ClientError, NoCredentialsError
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class S3Handler:
    def __init__(self):
        try:
            self.s3_client = boto3.client('s3',
                            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                            region_name=settings.AWS_REGION)
            self.bucket_name = settings.S3_BUCKET_NAME
            logger.info(f"S3 client initialized for bucket {self.bucket_name}")
        except NoCredentialsError:
            logger.error("AWS credentials not available")
            raise

    def verify_file_exists(self, key:str)->bool:
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError:
            return False

    def get_file_size(self, key:str)->int:
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return  response['ContentLength']
        except ClientError:
            logger.error(f"file {key} does not exist")
            raise

    def download_file(self, s3_key: str, local_path: Path) -> Path:
        """Download file from S3 with progress bar"""

        # Verify file exists
        if not self.verify_file_exists(s3_key):
            raise FileNotFoundError(f"File not found in S3: {s3_key}")

        # Create directory if doesn't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Get file size for progress bar
        file_size = self.get_file_size(s3_key)
        file_size_gb = file_size / (1024 ** 3)

        logger.info(f"Downloading {s3_key} ({file_size_gb:.2f} GB)...")
        logger.info(f"Destination: {local_path}")

        # Download with progress tracking
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            def progress_callback(bytes_transferred):
                pbar.update(bytes_transferred)

            try:
                self.s3_client.download_file(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Filename=str(local_path),
                    Callback=progress_callback
                )
            except ClientError as e:
                logger.error(f"Download failed: {e}")
                raise

        logger.info(f"✅ Download complete: {local_path}")
        return local_path

    def stream_download(self, s3_key: str, local_path: Path, chunk_size: int = 8192 * 1024) -> Path:
        """Stream download for very large files (memory efficient)"""

        if not self.verify_file_exists(s3_key):
            raise FileNotFoundError(f"File not found in S3: {s3_key}")

        local_path.parent.mkdir(parents=True, exist_ok=True)

        file_size = self.get_file_size(s3_key)
        file_size_gb = file_size / (1024 ** 3)

        logger.info(f"Streaming download {s3_key} ({file_size_gb:.2f} GB)...")

        try:
            s3_object = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)

            with open(local_path, 'wb') as f:
                with tqdm(total=file_size, unit='B', unit_scale=True, desc="Streaming") as pbar:
                    for chunk in s3_object['Body'].iter_chunks(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(len(chunk))
        except ClientError as e:
            logger.error(f"Stream download failed: {e}")
            raise

        logger.info(f"✅ Stream download complete: {local_path}")
        return local_path