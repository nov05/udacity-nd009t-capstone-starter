## /opt/ml/processing/input/code/webdataset.py
## This script demonstrates how to convert image and metadata pairs stored in S3 into WebDataset format
import os
import io
import boto3
import argparse
import webdataset as wds


def get_s3_object_keys(bucket_name, prefix):
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if 'Contents' not in response:
        print(f"⚠️ No objects found with prefix '{prefix}' in bucket '{bucket_name}'.")
        return []
    return  [obj['Key'] for obj in response['Contents']]
    

def iterate_in_chunks(input_list, chunk_size=10):
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i:i+chunk_size]


def convert_dataset(image_keys, num_tar_files):
    # Create a tar file in memory and write WebDataset format
    tar_stream = io.BytesIO()
    with wds.TarWriter(tar_stream) as sink:
        for image_key in enumerate(image_keys):
            if not (image_key.endswith('.jpg') or image_key.endswith('.jpeg')):
                print(f"⚠️ Skipping non-image file: {image_key}")
                continue
            base_name = os.path.splitext(image_key)[0]
            try:  # Ensure the corresponding JSON file exists
                with open(f's://{input_bucket}/{input_prefix_metadata}{base_name}.json', "rb") as f:
                    metadata_data = f.read()
            except Exception as e:
                print(f"⚠️ Skipping image '{image_key}' due to error: {e}")
                continue
            with open(f's://{input_bucket}/{image_key}', "rb") as f:
                image_data = f.read()
            # Save as WebDataset sample
            sink.write({
                "__key__": f"{base_name}",
                "image": image_data,
                "metadata": metadata_data
            })
    # Once the tar file is in memory, upload it back to S3
    tar_stream.seek(0)
    s3_client.upload_fileobj(tar_stream, output_bucket, f'{output_prefix}data_{num_tar_files}.tar')


def main():
    MAX_TAR_FILES = 2  ## Maximum number of tar files to create
    KEYS_PER_TAR = 10  ## Number of keys to process per tar file
    image_keys = get_s3_object_keys(input_bucket, input_prefix_images)
    num_tar_files = 0 
    for image_keys in iterate_in_chunks(image_keys, KEYS_PER_TAR):
        print("👉 image_keys_chunk:", image_keys)
        convert_dataset(image_keys, num_tar_files)
        num_tar_files += 1
        if num_tar_files >= MAX_TAR_FILES:
            break
    

if __name__ == "__main__":

    ## Verify WebDataset import
    print("👉 WebDataset dir:", dir(wds))

    parser = argparse.ArgumentParser()
    parser.add_argument('--SM_INPUT_BUCKET', type=str)
    parser.add_argument('--SM_INPUT_PREFIX_IMAGES', type=str)
    parser.add_argument('--SM_INPUT_PREFIX_METADATA', type=str)
    parser.add_argument('--SM_OUTPUT_BUCKET', type=str)
    parser.add_argument('--SM_OUTPUT_PREFIX', type=str)
    args = parser.parse_args()
    input_bucket = args.SM_INPUT_BUCKET
    input_prefix_images = args.SM_INPUT_PREFIX_IMAGES
    input_prefix_metadata = args.SM_INPUT_PREFIX_METADATA
    output_bucket = args.SM_OUTPUT_BUCKET
    output_prefix = args.SM_OUTPUT_PREFIX

    # Set up S3
    s3_client = boto3.client('s3')

    main()