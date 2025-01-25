* This markdown file is created by nov05 on 2025-01-25   

# **Input data serialization**  

**ChatGPT:**   

To optimize the read access pattern for 500K images, each around 100 KB, you can serialize the smaller image files into larger containers using formats like **TFRecord** (for TensorFlow) or **HDF5** (for general purposes). This helps group many small files into a few larger ones, reducing the overhead of opening and closing many individual files and improving I/O performance.

TFRecord is more suitable if you are working within the TensorFlow ecosystem, while HDF5 is more general-purpose and works well in most machine learning frameworks. Both methods reduce file system overhead and allow for faster sequential reads, especially when training models with large datasets.  

These approaches help in batch processing and faster I/O when you have many small images.


# **TFRecord code example**

```python
import tensorflow as tf
import os

def serialize_example(image_string, label):
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_tfrecord(images_path, labels, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for image_path, label in zip(images_path, labels):
            image_string = open(image_path, 'rb').read()
            tf_example = serialize_example(image_string, label)
            writer.write(tf_example)

def read_tfrecord(tfrecord_file):
    raw_image_dataset = tf.data.TFRecordDataset(tfrecord_file)

    # Define the features to parse
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    parsed_image_dataset = raw_image_dataset.map(_parse_function)
    return parsed_image_dataset

# Usage
images_path = ["path_to_image1.jpg", "path_to_image2.jpg"]
labels = [0, 1]  # Example labels for the images
write_tfrecord(images_path, labels, "images.tfrecord")

# Reading the TFRecord
dataset = read_tfrecord("images.tfrecord")
for record in dataset:
    image = record['image'].numpy()
    label = record['label'].numpy()
    # You can now process the image further
```

# **HDF5 code example**  

```python   
import h5py
import numpy as np
import cv2  # Assuming you're using OpenCV for image handling

def create_hdf5(images_path, labels, output_file):
    # Assuming all images have the same size, e.g., (100, 100, 3)
    img_shape = (100, 100, 3)
    
    with h5py.File(output_file, 'w') as f:
        img_dataset = f.create_dataset('images', (len(images_path), *img_shape), dtype='uint8')
        label_dataset = f.create_dataset('labels', (len(labels),), dtype='int')

        for i, (image_path, label) in enumerate(zip(images_path, labels)):
            image = cv2.imread(image_path)
            img_dataset[i] = image
            label_dataset[i] = label

def read_hdf5(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        images = f['images'][:]
        labels = f['labels'][:]
        return images, labels

# Usage
images_path = ["path_to_image1.jpg", "path_to_image2.jpg"]
labels = [0, 1]  # Example labels for the images
create_hdf5(images_path, labels, "images.hdf5")

# Reading the HDF5 file
images, labels = read_hdf5("images.hdf5")
```   