# This script demonstrates how to convert image and metadata pairs stored in S3 into WebDataset format
import boto3
import io
import webdataset as wds
from s3fs import S3FileSystem

# Set up S3
s3 = boto3.client('s3')
s3fs = S3FileSystem()

# Define your input/output buckets and prefixes
input_bucket = 'your-input-bucket'
input_images_prefix = 'images/'
input_metadata_prefix = 'metadata/'
output_bucket = 'your-output-bucket'
output_prefix = 'webdataset/'

def convert_to_webdataset(num_files):
    # Define where the input files are stored in S3
    input_images_s3_path = f's3://{input_bucket}/{input_images_prefix}'
    input_metadata_s3_path = f's3://{input_bucket}/{input_metadata_prefix}'

    # Create a tar file in memory and write WebDataset format
    tar_stream = io.BytesIO()

    # Example: Iterate through the image and metadata pairs (using s3fs to stream)
    with s3fs.open(f'{input_images_s3_path}image1.jpg', 'rb') as image_file, \
         s3fs.open(f'{input_metadata_s3_path}metadata1.json', 'rb') as metadata_file:
        
        # Read and convert data into WebDataset tar format
        with wds.TarWriter(tar_stream) as sink:
            for i in range(num_files):
                # Assuming image_file and metadata_file are pairs for the same data
                image_data = image_file.read()
                metadata_data = metadata_file.read()

                # Save as WebDataset sample with "__key__", "jpg", and "json"
                sink.write({
                    "__key__": f"sample{i}",
                    "jpg": image_data,
                    "json": metadata_data
                })

    # Once the tar file is in memory, upload it back to S3
    tar_stream.seek(0)
    s3.upload_fileobj(tar_stream, output_bucket, f'{output_prefix}data_{i}.tar')

def main():
    num_files = 60  # Adjust this based on the actual number of image-metadata pairs
    convert_to_webdataset(num_files)

if __name__ == "__main__":
    main()