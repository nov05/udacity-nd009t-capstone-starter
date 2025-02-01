## /opt/ml/processing/input/code/webdataset.py
## This script shuffle then splits dataset to train, val, and test, 
## and converts image and metadata pairs stored in S3 into WebDataset format
import os
import boto3
import argparse
import webdataset as wds
import random
import json
import glob



def split_dataset(file_list, ratio=[0.7, 0.15, 0.15]):
    # Split dataset into train, validation, and test sets (70%, 15%, 15%)
    l = len(file_list)
    train_size = int(l*ratio[0])
    val_size = int(l*ratio[1])
    test_size = l - train_size - val_size
    return file_list[:train_size], \
           file_list[train_size:train_size+val_size], \
           file_list[-test_size:]


def get_file_list(s3_uri):
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    response = s3_client.get_object(Bucket=bucket, Key=key)
    json_content = json.loads(response["Body"].read().decode("utf-8"))
    file_list = []
    for label, file_name_list in json_content.items():
        for file_name in file_name_list:
            file_list.append((file_name.split("/")[-1].split(".")[0], label))  # (image_id, label)
    random.shuffle(file_list)
    print(f"🟢 File list successfully loaded from {s3_uri}\n"
          f"    Total number of image files: {len(file_list)}")
    return file_list  


def read_s3_file(bucket, key):
    # Read the file from S3
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return response['Body'].read()


def convert_dataset(type_prefix,  ## e.g. "train/"
                    file_list, 
                    maxcount=1000):  ## number of items per shard
    shard_prefix = type_prefix[:-1] + "-shard-"  ## e.g. file name: "train-shard-000000.tar"
    with wds.ShardWriter(f"{shard_prefix}%06d.tar", maxcount=maxcount) as sink:
        for image_id,label in file_list:
            image_key = f'{input_prefix_images}{image_id}.jpg'
            try:  # Ensure the corresponding JSON file exists
                image_data = read_s3_file(input_bucket, image_key)
            except Exception as e:
                print(f"⚠️ Skipping image '{image_key}' due to error: {e}")
                continue
            # Save as WebDataset sample
            sink.write({
                "__key__": f"{image_id}",
                "image": image_data,
                "label": label,
            })
    # Upload the shard files to S3, then delete them from the processor instance locally
    shard_list = glob.glob(f"{shard_prefix}*.tar")
    for shard_file in shard_list:
        base_name = os.path.basename(shard_file)
        s3_key = os.path.join(output_prefix, type_prefix, base_name)
        s3_client.upload_file(shard_file, output_bucket, s3_key)
        if os.path.exists(shard_file):
            os.remove(shard_file)
    print(f"🟢 Successfully uploaded shard files to "
          f"s3://{output_bucket}/{output_prefix}{type_prefix}:\n"
          f"    {shard_list}")


def main():
    FILE_LIST_URI = "s3://p5-amazon-bin-images/file_list.json"  ## total number: 10441
    KEYS_PER_TAR = 1000  ## Number of keys to process per tar file
    file_list = get_file_list(FILE_LIST_URI) 
    for type_prefix, file_list in zip(['train/', 'val/', 'test/'], 
                                      split_dataset(file_list)):
        convert_dataset(type_prefix, 
                        file_list, 
                        maxcount=KEYS_PER_TAR)



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--SM_INPUT_BUCKET', type=str)
    parser.add_argument('--SM_INPUT_PREFIX_IMAGES', type=str)
    parser.add_argument('--SM_INPUT_PREFIX_METADATA', type=str)
    parser.add_argument('--SM_OUTPUT_BUCKET', type=str)
    parser.add_argument('--SM_OUTPUT_PREFIX', type=str)
    args = parser.parse_args()
    input_bucket = args.SM_INPUT_BUCKET
    input_prefix_images = args.SM_INPUT_PREFIX_IMAGES  ## images/
    input_prefix_metadata = args.SM_INPUT_PREFIX_METADATA  ## metadata/
    output_bucket = args.SM_OUTPUT_BUCKET
    output_prefix = args.SM_OUTPUT_PREFIX  ## webdataset/

    random.seed(42)
    s3_client = boto3.client('s3')

    print("Starting data processing...")
    main()