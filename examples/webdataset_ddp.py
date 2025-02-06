## test code SageMaker DDP + Torch DDP
import webdataset as wds
import matplotlib.pyplot as plt
from PIL import Image
import io
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, IterableDataset
# from torch.utils.data.distributed import DistributedSampler  ## ⚠️
import numpy as np
import time

import multiprocessing as mp
print("👉 Multiprocessing start method:", mp.get_start_method(allow_none=True))
# if mp.get_start_method(allow_none=True) != 'spawn':
#     torch.multiprocessing.set_start_method('spawn')  

def key_transform(x):
    return int(x)

class image_transform:
    def __call__(self, x):
        return Image.open(io.BytesIO(x))
    
train_transform = transforms.Compose([
    image_transform(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
])  

def label_transform(x):
    return torch.tensor(int(x.decode()))

class WebDatasetDDP(IterableDataset):
    def __init__(self,
                 path, 
                 *args, 
                 world_size=1, 
                 rank=0,  
                 shuffle_buffer_size=1000,
                 num_samples=0,
                 shardshuffle=True,
                 empty_check=False,
                 key_transform=None,
                 train_transform=None,
                 label_transform=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = (
            wds.WebDataset(
                path, 
                shardshuffle=shardshuffle,
                # nodesplitter=wds.split_by_worker,
                empty_check=empty_check, 
            )
            .shuffle(shuffle_buffer_size)  # Shuffle dataset 
            ## The tuple names have to be the same with the WebDataset keys
            ## check the "scripts_process/*convert_to_webdataset*.py" files
            .to_tuple("__key__", "image", "label")  ## Tuple of image and label
            .map_tuple(
                key_transform,
                train_transform,  # Apply the train transforms to the image
                ## lambda function can't not be pickled, hence cause error when num_workers>1 
                label_transform,  
            )
        )
        self.world_size = world_size
        self.rank = rank
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples
    
    def __iter__(self): 
        for key,image,label in self.dataset:
            if key%self.world_size == self.rank:  ## Ensure each GPU gets different data
                yield (image, label)

def collate_fn(batch):
    images, labels = zip(*batch)
    images = [torch.tensor(np.array(image)) for image in images] 
    # Stack the images into a single tensor (this assumes the images have the same size)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels

def main():
    s3_uri = "s3://p5-amazon-bin-images/webdataset/train/train-shard-{000000..000000}.tar"
    path = f"pipe:aws s3 cp {s3_uri} -"  ## write to standard output (stdout)
    train_dataset = (
        WebDatasetDDP(
            path, 
            world_size=2, 
            rank=0, 
            shuffle_buffer_size=1000,
            num_samples=1000,
            shardshuffle=True,
            empty_check=False,
            key_transform=key_transform,
            train_transform=train_transform,
            label_transform=label_transform,
        )
    ) 
    # ## ⚠️ DistributedSampler doesn't work with IterableDataset
    # train_sampler = DistributedSampler(
    #     train_dataset,
    #     num_replicas=2,  ## number of GPUs
    #     rank=0,  ## global rank
    #     shuffle=False,
    #     drop_last=True,
    # )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=int(256/2),  ## 2 GPUs 
        shuffle=False,  ## Don't shuffle for Distributed Data Parallel (DDP)  
        # sampler=train_sampler, # ⚠️ Distributed Sampler causes error
        # num_workers=2, # ⚠️ Cause error in windows
        # persistent_workers=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    print(f"train_loader length: {len(train_loader)}")
    print(f"train_loader.dataset length: {len(train_loader.dataset)}")
    for batch_idx, batch in enumerate(train_loader):
        print(batch_idx, len(batch[0]), len(batch[1]))
        
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"👉 Execution Time: {end_time-start_time} seconds")

## On Windows...
## 8 workers, 31.25 s
## 4 workers, 29.26 s
## 2 workers, 29.40 s
## 1 workers, 20.52 s

## $ cd examples
## $ python -m webdataset_ddp
