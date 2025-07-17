import boto3

def upload_to_s3(bucket, source, dest):
    s3 = boto3.client('s3')
    s3.upload_file(source, bucket, dest)
    print(f"Uploaded {source} to s3://{bucket}/{dest}")

def download_from_s3(bucket, source, dest):
    s3 = boto3.client('s3')
    s3.download_file(bucket, source, dest)
    print(f"Downloaded s3://{bucket}/{source} to {dest}")