## https://sagemaker.readthedocs.io/en/stable/api/training/sdp_versions/v1.0.0/smd_data_parallel_pytorch.html
## https://github.com/webdataset/webdataset/blob/90346059ec6a64a950c37c252e38db64db00de0b/examples/train-resnet50-multiray-wds.ipynb

## TODO: Import your dependencies.
## For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
# from torchvision import datasets
from torchvision import transforms
## For streamed training data, we can't access the full dataset, 
## hence use pre-calculated or esitmated class weights
# from sklearn.utils.class_weight import compute_class_weight
import os, io
from pprint import pprint
import argparse
import wandb

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow truncated images

## Move the import to the main check part of this script
## TODO: Import dependencies for Debugging andd Profiling
# ====================================#
# 1. Import SMDebug framework class.  #
# ====================================#
# import smdebug.pytorch as smd

## Dataset
import webdataset as wds
## PyTorch Distributed Data Parallel (DDP) 
# import torch.multiprocessing as mp
from torch.utils.data import DataLoader, IterableDataset
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group
## SageMaker distributed data parallel library PyTorch APIs
import smdistributed.dataparallel.torch.distributed as dist
from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
dist.init_process_group(backend="smddp")  ## SageMaker DDP, replacing "nccl"



def get_shard_number(path):
    ## e.g. "train/train-shard-{000000..000001}.tar", 2 shards
    start, _, end = path.split('{')[-1].split('}')[0].split('.')
    if end is None:
        return int(start)
    else:
        return int(end)-int(start)+1

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
    ## Original lables are (1,2,3,4,5)
    ## Convert to (0,1,2,3,4)
    return torch.tensor(int(x.decode())-1, dtype=torch.int64)



## WebDataset class inherits from IterableDataset class
class WebDatasetDDP(IterableDataset):
    def __init__(self,
                 path,  
                 num_samples=0,
                 world_size=1, 
                 rank=0,  
                 no_shuffle=False,
                #  shardshuffle=True,
                 shuffle_shard_size=100,
                 nodesplitter=wds.split_by_node,
                 key_transform=None,
                 train_transform=None,
                 label_transform=None,
                 shuffle_sample_size=1000,
                 batch_size=256,
                #  empty_check=False,
                ):
        super().__init__()
        self.dataset = (
## WebDataset
## https://github.com/webdataset/webdataset?tab=readme-ov-file#the-webdataset-library
            # wds.WebDataset(
            #     path, 
            #     resample=True,
            #     shardshuffle=shardshuffle,
            #     ## Official doc: add wds.split_by_node here if you are using multiple nodes
            #     nodesplitter=wds.split_by_node, 
            #     ## Or "ValueError: you need to add an explicit nodesplitter 
            #     ## to your input pipeline for multi-node training"
            #     #nodesplitter=wds.split_by_worker,
            #     empty_check=empty_check, 
            # )
            # .shuffle(shuffle_buffer_size)  # Shuffle dataset 
            # ## The tuple names have to be the same with the WebDataset keys
            # ## check the "scripts_process/*convert_to_webdataset*.py" files
            # .to_tuple("__key__", "image", "label")  ## Tuple of image and label
            # .map_tuple(
            #     key_transform,
            #     train_transform,  # Apply the train transforms to the image
            #     ## lambda function can't not be pickled, hence cause error when num_workers>1 
            #     label_transform,  
            # )
## WebDataset pipeline
## https://github.com/webdataset/webdataset?tab=readme-ov-file#the-webdataset-pipeline-api
            wds.DataPipeline(
                wds.SimpleShardList(path),
                # at this point we have an iterator over all the shards
                wds.shuffle(shuffle_shard_size) if not no_shuffle else None,
                ## nodesplitter Options: wds.single_node_only, wds.split_by_node, 
                ##     wds.split_by_worker, split_by_node_worker, None
                ## add wds.split_by_node here if you are using multiple nodes
                ## "worker" is not used by SMDDP
                nodesplitter,
                # at this point, we have an iterator over the shards assigned to each worker
                wds.tarfile_to_samples(),
                # this shuffles the samples in memory
                wds.shuffle(shuffle_sample_size) if not no_shuffle else None,
                # this decodes the images and json
                # wds.decode("pil"),
                wds.to_tuple("__key__", "image", "label"),
                # wds.map(preprocess),
                wds.map_tuple(
                    key_transform,
                    train_transform, 
                    label_transform,  
                ),
                wds.shuffle(shuffle_sample_size) if not no_shuffle else None,
                wds.batched(batch_size),
            )
        )
        self.world_size = world_size
        self.rank = rank
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples
    
    def __iter__(self):  ## custome iterator
        for key,image,label in self.dataset:  ## Use dataset keys to distribute data
            ## ⚠️ here needs a fix
            if key%self.world_size == self.rank:  ## Ensure each GPU gets different data
                yield (image, label)



def collate_fn(batch):
    images, labels = zip(*batch)
    ## Stack the images into a single tensor
    ## This assumes the images have the same size
    images = torch.stack(images)
    labels = torch.stack(labels)
    return images, labels



## SageMaker DDP
def average_gradients(model):
    for param in model.parameters():
        dist.all_reduce(
            param, 
            op=dist.ReduceOp.SUM, 
            group=dist.group.WORLD, 
            async_op=False)
        param.data /= float(dist.get_world_size())



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('⚠️ Boolean value expected')
    


def str2dict(s):
    if not s:
        return {}
    try:
        # Split the string into key-value pairs and convert to a dictionary
        return {int(k):float(v) for k,v in (item.split('=') for item in s.split(','))}
    except Exception as e:
        print(e)
        raise argparse.ArgumentTypeError(f"⚠️ Invalid dictionary: {s}")
    


class Config:
    def __init__(self):
        self.wandb = False
        self.debug = False



class Task:
    def __init__(self):
        self.config = Config()
        self.hook = None  ## SageMaker debugger hook
        self.model = None



## For wandb logging
class StepCounter:
    def __init__(self):
        self.total_steps = 0
    
    def __call__(self):
        self.total_steps += 1
    
    def reset(self):
        self.total_steps = 0



class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0



## PyTorch DDP
# def ddp_setup(rank: int, world_size: int):
#     """
#     Args:
#         rank: Unique identifier of each process
#         world_size: Total number of processes
#     """
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "12355"
#     torch.cuda.set_device(rank)
#     init_process_group(backend="nccl",  
#                        rank=rank, 
#                        world_size=world_size)



def train(task):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    # =================================================#
    # 2. Set the SMDebug hook for the training phase. #
    # =================================================#
    if task.config.debug: 
        task.hook.set_mode(smd.modes.TRAIN)
    ## Set the model to training mode
    task.model.train()
    num_samples = 0.
    for batch_idx, (data, target) in enumerate(task.train_loader):  ## Torch loader & WebDataset loader
        num_samples += len(target)
        task.step_counter()
        data, target = data.to(task.config.device), target.to(task.config.device)  ## images, labels
        task.optimizer.zero_grad()
        output = task.model(data)  
        loss = task.train_criterion(output, target)
        if task.config.wandb and dist.get_rank()==0:
            wandb.log({f"Rank {dist.get_rank()}, train_loss": loss.item()}, 
                      step=task.step_counter.total_steps)
        loss.backward()
        average_gradients(task.model)  ## SageMaker DDP
        task.optimizer.step()
        torch.cuda.empty_cache()
        if batch_idx%1 == 0:  ## Print every n batches
            ## Torch loader without DDP
            # print(
            #     "Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f}".format(
            #         task.current_epoch,
            #         batch_idx * len(data),
            #         len(task.train_loader.dataset),
            #         100.0 * batch_idx / len(task.train_loader),
            #         loss.item(),
            #     )
            # )
            ## WebDataset loader with DDP
            print(
                "🔹 Train Epoch {:.0f}, Rank {:.0f}: [{:.0f}/{:.0f} ({:.0f}%)], Loss: {:.6f}".format(
                    task.current_epoch,                                      ## 1. current epoch
                    dist.get_rank(),                                         ## 2. global rank
                    num_samples,                                             ## 3. samples that have been used
                    task.config.train_data_size / dist.get_world_size(),     ## 4. total number of samples
                                                                             ## 5. progress within the epoch
                    100.0*num_samples*dist.get_world_size() / task.config.train_data_size,       
                    loss.item(),                                             ## 6. loss value
                )
            )



def eval(task, phase='val'):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    NOTE: 1. val and test are not distributed. 
          2. len(data_loader.dataset) is technically unknown,
             unless it is pre-set as hyperparameter
    '''
    # ===================================================#
    # 3. Set the SMDebug hook for the validation phase.  #
    # ===================================================#
    if task.config.debug and task.hook: 
        task.hook.set_mode(smd.modes.EVAL)
    task.model.eval()
    test_loss = 0.
    correct = 0.
    num_samples = 0.
    data_loader = task.val_loader if phase=='eval' else task.test_loader
    with torch.no_grad():
        for data, target in data_loader:  ## PyTorch loader & WebDataset loader
            num_samples += len(target)
            data, target = data.to(task.config.device), target.to(task.config.device)
            output = task.model(data)
            test_loss += task.val_criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= num_samples # len(data_loader.dataset) 
    if phase=='val': 
        task.early_stopping(test_loss)
        if task.config.wandb:
            wandb.log({f"{phase}_loss_epoch": test_loss}, 
                      step=task.step_counter.total_steps)
    accuracy = 100.*correct/num_samples # len(data_loader.dataset)
    print(
        "\n👉 {}: Average loss: {:.4f}, Accuracy: {:.0f}/{:.0f} ({:.2f}%)\n".format(
            phase.upper(),
            test_loss, 
            correct, 
            num_samples, # len(data_loader.dataset),  ## test data size, task.config.test_data_size
            accuracy
        )
    )
    if phase=='val' and task.config.wandb:
        wandb.log(
            {f"{phase}_accuracy_epoch": accuracy}, 
            step=task.step_counter.total_steps
        )



def create_net(task):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    task.model = getattr(torchvision.models, task.config.model_arch)(
        # weights='IMAGENET1K_V1',  ## Load pretrained weights
    ) 
    task.model.fc = nn.Linear(
        task.model.fc.in_features, 
        task.config.num_classes)  # Adjust for the number of classes
    torch.nn.init.kaiming_normal_(task.model.fc.weight)  # Initialize new layers
    task.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(task.model)  ## PyTorch DDP
    ## Wrap model with smdistributed.dataparallel's DistributedDataParallel
    task.model = DDP(task.model.to(task.config.device), 
                     device_ids=[torch.cuda.current_device()])  ## single-device Torch module 
    print(f"👉 Rank {dist.get_rank()}: "
          f"Model {task.config.model_arch} has been created successfully.") 



def save(task):
    '''
    Save the model to the model_dir
    '''
    task.model.eval()
    path = os.path.join(task.config.model_dir, 'model.pth')
    ## save model weights only
    with open(path, 'wb') as f:
        torch.save(task.model.state_dict(), f)
    # torch.save(task.model.module.state_dict(), path)  ## SageMaker Model Parallel? SMDDP

    ## Please ensure model is saved using torchscript when necessary.
    ## https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
    '''
    ## If your model is simple and has a straightforward forward pass, use torch.jit.trace
    example_input = torch.randn(1, 3, 224, 224)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(path)
    '''
    '''
    ## If your model has dynamic control flow (like if statements based on input), 
    ## use torch.jit.script
    scripted_model = torch.jit.script(model)
    scripted_model.save(path) 
    '''
    print(f"👉 Model saved at '{path}'")



def main(task):  ## rank is auto-allocated by DDP when calling mp.spawn
    '''
    Train, eval, test, and save the model
    '''
    task.step_counter = StepCounter()
    task.early_stopping = EarlyStopping(task.config.early_stopping_patience)
    task.config.device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        if task.config.use_cuda else "cpu"
    )
    print(f"👉 Device: {task.config.device}, "
          f"Rank: {dist.get_rank()}, "  ## SMDDP
          f"Local rank: {dist.get_local_rank()}")  ## SMDDP
    task.config.num_cpu = os.cpu_count()  ## for data loaders

    ## before initializing the group process, call set_device, 
    ## which sets the default GPU for each process. This is important 
    ## to prevent hangs or excessive memory utilization on GPU:0.
    # ddp_setup(rank, world_size)  ## DDP

    train_transform = transforms.Compose([
        image_transform(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=0.2, 
                               contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])  
    val_transform = transforms.Compose([
        image_transform(),
        transforms.Resize((224, 224)),  ## default: interpolation=InterpolationMode.BILINEAR
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ]) 

    ## Replace TorchVision.datasets.ImageFolder() with WebDataset.dataset() pipe
    # train_dataset = datasets.ImageFolder(task.config.train, transform=train_transform)
    # val_dataset = datasets.ImageFolder(task.config.validation, transform=val_transform)
    # test_dataset = datasets.ImageFolder(task.config.test, transform=val_transform)

    ## For data distributed training, use torch.utils.data.DistributedSampler or WebDataset? 
    path = f"pipe:aws s3 cp {task.config.train_data_path} -"
    train_dataset = (
        # WebDatasetDDP(
        #     path, 
        #     num_samples=task.config.train_data_size,
        #     world_size=dist.get_world_size(), 
        #     rank=dist.get_rank(), 
        #     nodesplitter=wds.split_by_node,
        #     shuffle_sample_size=1000,
        #     key_transform=key_transform,
        #     train_transform=train_transform,
        #     label_transform=label_transform,
        #     batch_size=task.config.batch_size,
        # )
        wds.DataPipeline(
            wds.SimpleShardList(path),
            # at this point we have an iterator over all the shards
            wds.shuffle(1000),
            ## nodesplitter Options: wds.single_node_only, wds.split_by_node, 
            ##                       wds.split_by_worker, split_by_node_worker, None
            ## use wds.split_by_node here if you are using multiple nodes
            ## "worker" values don't exist with SMDDP
            wds.split_by_node,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(),
            # this shuffles the samples in memory
            wds.shuffle(1000),
            # this decodes the key, image and json
            # wds.to_tuple("__key__", "image", "label"),
            wds.to_tuple('image', 'label'),
            wds.map_tuple(
                # key_transform,
                train_transform, 
                label_transform,  
            ),
            wds.shuffle(1000),
            wds.batched(task.config.batch_size),
        )
    ) 
    path = f"pipe:aws s3 cp {task.config.val_data_path} -"
    val_dataset = (
        # WebDatasetDDP(
        #     path, 
        #     num_samples=task.config.val_data_size,
        #     world_size=dist.get_world_size(), 
        #     rank=dist.get_rank(), 
        #     no_shuffle=True,
        #     nodesplitter=None,
        #     key_transform=key_transform,
        #     train_transform=val_transform,
        #     label_transform=label_transform,  
        #     batch_size=task.config.batch_size,               
        # )  
        wds.DataPipeline(
            wds.SimpleShardList(path),
            wds.tarfile_to_samples(),
            # wds.to_tuple("__key__", "image", "label"),
            wds.to_tuple('image', 'label'), 
            wds.map_tuple(
                # key_transform,
                val_transform, 
                label_transform,  
            ),
            wds.batched(task.config.batch_size),
        ) 
    )
    path = f"pipe:aws s3 cp {task.config.test_data_path} -"
    test_dataset = (
        # WebDatasetDDP(
        #     path, 
        #     num_samples=task.config.test_data_size,
        #     world_size=dist.get_world_size(), 
        #     rank=dist.get_rank(), 
        #     no_shuffle=True,
        #     nodesplitter=None,
        #     key_transform=key_transform,
        #     train_transform=val_transform,
        #     label_transform=label_transform,   
        #     batch_size=task.config.batch_size,              
        # )  
        wds.DataPipeline(
            wds.SimpleShardList(path),
            wds.tarfile_to_samples(),
            # wds.to_tuple("__key__", "image", "label"),
            wds.to_tuple('image', 'label'),
            wds.map_tuple(
                # key_transform,
                val_transform, 
                label_transform,  
            ),
            wds.batched(task.config.batch_size),
        ) 
    )
 
    ## Handle class imbalance. class weights will be used in the loss functions.
    ## train_dataset is an instance of TorchVision.datasets.ImageFolder().
    ## class_weights is an instance of <class 'numpy.ndarray'>.
    # class_weights = compute_class_weight(
    #     class_weight='balanced', 
    #     classes=np.unique(train_dataset.cls),   
    #     y=train_dataset.cls)
    ## Use pre-calculated class weights if the dataset is very large.
    classes = np.unique(list(task.config.class_weights_dict.keys()))   ## It has to be sorted.
    task.config.num_classes = len(classes)  ## get number of total classes for net creation
    class_weights = [task.config.class_weights_dict[k] for k in classes]
    class_weights = torch.tensor(
        class_weights, 
        dtype=torch.float32).to(task.config.device)
    
    # ## SMDDP: set num_replicas and rank in Torch DistributedSampler
    # train_sampler = DistributedSampler(  ## ⚠️ doesn't work with WebDataset
    #     train_dataset,
    #     num_replicas=dist.get_world_size(),
    #     rank=dist.get_rank(),
    #     shuffle=False,
    # )
    
    ## Torch dataloader (incompatible with generic IterableDataset)
    # task.train_loader = DataLoader(
    #     train_dataset, 
    #     batch_size=task.config.batch_size_ddp, 
    #     shuffle=False,  ## Don't shuffle for Distributed Data Parallel (DDP)  
    #     # sampler=train_sampler, # ⚠️ Distributed Sampler + WebDataset causes error
    #     num_workers=task.config.num_cpu,
    #     persistent_workers=True,
    #     pin_memory=True,
    #     collate_fn=collate_fn,
    # )
    # task.val_loader = DataLoader(
    #     val_dataset, 
    #     batch_size=task.config.batch_size, 
    #     shuffle=False,   ## Don't shuffle for eval anyway
    #     ## no DDP sampler
    #     num_workers=task.config.num_cpu,
    #     persistent_workers=True,
    #     pin_memory=True,
    #     collate_fn=collate_fn,
    # )
    # task.test_loader = DataLoader(
    #     test_dataset, 
    #     batch_size=task.config.batch_size, 
    #     shuffle=False,  ## Don't shuffle for eval anyway
    #     # no DDP sampler
    #     num_workers=task.config.num_cpu,
    #     persistent_workers=True,
    #     pin_memory=True,
    #     collate_fn=collate_fn,
    # )
    ## WebDataset dataloader
    num_batches = task.config.train_data_size // (task.config.batch_size * dist.get_world_size())
    task.train_loader = (
        wds.WebLoader(
            train_dataset, 
            batch_size=None, 
            num_workers=task.config.num_cpu,
        ).unbatched()
        .shuffle(1000)
        .batched(task.config.batch_size)
        ## A resampled dataset is infinite size, but we can recreate a fixed epoch length.
        .with_epoch(num_batches)   
    )
    num_batches = task.config.val_data_size // task.config.batch_size
    task.val_loader = (
        wds.WebLoader(
            val_dataset, 
            batch_size=None, 
            num_workers=task.config.num_cpu,
        )
        .with_epoch(num_batches)  
    )
    num_batches = task.config.test_data_size // task.config.batch_size
    task.test_loader = (
        wds.WebLoader(
            test_dataset, 
            batch_size=None, 
            num_workers=task.config.num_cpu,
        )
        .with_epoch(num_batches) 
    )
    

    ## TODO: Initialize a model by calling the net function
    create_net(task)
    ## SMDDP: Pin each GPU to a single distributed data parallel library process.
    torch.cuda.set_device(dist.get_local_rank())
    task.model.cuda(dist.get_local_rank())


    # ======================================================#
    # 4. Register the SMDebug hook to save output tensors.  #
    # ======================================================#
    if task.config.debug is True:
        task.hook = smd.Hook.create_from_json_file()
        task.hook.register_hook(task.model)  
    ## TODO: Create your loss and optimizer
    # criterion = nn.CrossEntropyLoss() 
    task.train_criterion = nn.CrossEntropyLoss(weight=class_weights)  # loss per step
    task.val_criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="sum")  ## loss per epoch
    if task.config.debug and task.hook is not None:
        task.hook.register_loss(task.train_criterion)
        task.hook.register_loss(task.val_criterion)
    task.optimizer = optim.AdamW(
        task.model.parameters(), 
        lr=task.config.opt_learning_rate * dist.get_world_size(),  ## SMDDP
        weight_decay=task.config.opt_weight_decay,
    )  
    task.scheduler = optim.lr_scheduler.StepLR(
        task.optimizer, 
        step_size=task.config.lr_sched_step_size, 
        gamma=task.config.lr_sched_gamma)  # Reduce LR every 6 epochs by 0.5
    '''
    TODO: Call the train function to start training your model
          Remember that you will need to set up a way to get training data from S3
    '''
    # ===========================================================#
    # 5. Pass the SMDebug hook to the train and test functions.  #
    # ===========================================================#
    for epoch in range(task.config.epochs):
        task.current_epoch = epoch
        if dist.get_rank()==0:
            print(f"👉 Train Epoch: {epoch}, "
                  f"Learning Rate: {task.optimizer.param_groups[0]['lr']}")
        # task.train_loader.sampler.set_epoch(epoch)  ## ⚠️ for Torch DDP
        train(task)
        if dist.get_rank()==0:
            eval(task, phase='val')
            if task.early_stopping.early_stop:
                print("⚠️ Early stopping")
                break
        task.scheduler.step()  ## Update learning rate after every epoch

    ## TODO: Test the model to see its accuracy
    if dist.get_rank()==0:
        print("🟢 Start testing...")
        eval(task, phase='test')

    ## TODO: Save the trained model
    if dist.get_rank()==0:  ## DDP only save one model
        save(task)

    ## According to the SMDDP document, there is no need to explicitly do this.
    # destroy_process_group()  ## PyTorch DDP



if __name__=='__main__':



    ## SageMaker DDP enviroment information
    if dist.is_initialized():
        print("🟢 SageMkaer DDP is initialized.")
    else:
        raise RuntimeError("⚠️ SageMaker DDP is not initialized.")
    print(f"👉 Total GPU count: {dist.get_world_size()}")
    ## Those are the env variables that are used by WebDataset splitting
    ## the rank of the worker node, and the rank of the GPU
    print(f"👉 Rank: {dist.get_rank()}, Local Rank: {dist.get_local_rank()}")  
    ## worker and number_workers cannot be found in SMDDP
    # if 'WORKER' in os.environ: print("👉 WORKER:", os.environ['WORKER'])  
    # if 'NUM_WORKERS' in os.environ: 
    #     print("👉 NUM_WORKERS:", os.environ['NUM_WORKERS']) 
    # else:
    #     worker_info = torch.utils.data.get_worker_info()
    #     if worker_info is not None:
    #         worker = worker_info.id
    #         num_workers = worker_info.num_workers
    #         print(f"👉 WOKER: {worker}, NUM_WORKERS: {num_workers}]") 

    parser=argparse.ArgumentParser()
    ## Hyperparameters passed by the SageMaker estimator
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--opt-learning-rate', type=float, default=1e-4)
    parser.add_argument('--opt-weight-decay', type=float, default=1e-4)  
    parser.add_argument('--lr-sched-step-size', type=int, default=6)  
    parser.add_argument('--lr-sched-gamma', type=float, default=0.5)  
    parser.add_argument('--early-stopping-patience', type=int, default=100)  
    parser.add_argument('--use-cuda', type=str2bool, default=True)
    ## Data, model, and output directories
    parser.add_argument('--model-arch', type=str, default='resnet34')
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    # parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    # parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    # parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    ## WebDataset paths
    parser.add_argument('--train-data-path', type=str, default='')
    parser.add_argument('--val-data-path', type=str, default='')
    parser.add_argument('--test-data-path', type=str, default='')
    ## training data info
    parser.add_argument('--train-data-size', type=int, default=0)
    parser.add_argument('--val-data-size', type=int, default=0)
    parser.add_argument('--test-data-size', type=int, default=0)
    parser.add_argument('--class-weights-dict', type=str2dict, default='')
    ## Others
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--wandb', type=str2bool, default=False)
    ## Retrieve arguments
    args, _ = parser.parse_known_args()
    task = Task()
    ## Use vars(args) to convert Namespace into a dictionary
    for key, value in vars(args).items():  
        ## Store arguments in the config object
        setattr(task.config, key, value)
    ## Get shard numbers
    task.config.train_shards = get_shard_number(task.config.train_data_path)
    task.config.val_shards = get_shard_number(task.config.val_data_path)
    task.config.test_shards = get_shard_number(task.config.test_data_path)
    if dist.get_rank()==0:
        print('👉 task.config:')
        pprint(task.config.__dict__)

    ## Initialize wandb
    if task.config.wandb and dist.get_rank()==0:
        task.wandb_run = wandb.init(
            ## set the wandb project where this run will be logged
            project="udacity-awsmle-resnet34-amazon-bin",
            config=args,
        )
    ## TODO: Import dependencies for Debugging andd Profiling
    # ====================================#
    # 1. Import SMDebug framework class.  #
    # ====================================#
    if task.config.debug:
        import smdebug.pytorch as smd
    ## Enables cuDNN's auto-tuner to find the best algorithm for the hardware
    torch.backends.cudnn.benchmark = True

    ## I'm not sure if SMDDP sets something like os.environ['WORLD_SIZE'], 
    ## or if I need to configure it manually using torch.cuda.device_count().
    # task.config.world_size = torch.cuda.device_count()  ## for DDP, the total number of GPUs
    # task.config.world_size = os.environ['WORLD_SIZE']

    ## I'm not sure if SageMaker Distributed Data Parallel (SMDDP) handles the spawning process, 
    ## or if I need to use mp.spawn() myself as instructed by PyTorch DDP.
    # mp.spawn(main,   ## main() in multiprocessing
    #          args=(task,),   ## rank is auto-allocated by PyTorch DDP when calling mp.spawn
    #          nprocs=task.config.world_size)
    main(task)

    ## Finish wandb run
    if task.config.wandb and dist.get_rank()==0:
        try:
            wandb.config.update(task.config.__dict__, 
                                allow_val_change=True)
        except Exception as e:
            print(f"⚠️ Updating wandb config failed: {e}")
        wandb.finish()