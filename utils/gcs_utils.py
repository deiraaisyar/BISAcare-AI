from google.cloud import storage

def upload_to_gcs(local_file_path: str, bucket_name: str, destination_blob_name: str) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file_path)
    blob.make_public()  # agar bisa diakses frontend
    return blob.public_url