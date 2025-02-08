## Created by nov05 on 2025-02-07
## 1. This script sets up distributed training using AWS SageMaker's Distributed Data Parallel (DDP) framework
##    and integrates with WebDataset for efficient data streaming from S3. 
## 2. We are also logging experiment metrics and configurations using Weights & Biases (wandb).
## 3. Early stopping is implemented.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import os, io
from pprint import pprint
from datetime import timedelta
import argparse
import wandb

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow truncated images

import webdataset as wds
import smdistributed.dataparallel.torch.distributed as dist
from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
dist.init_process_group(backend="smddp")  ## SageMaker DDP, replacing "nccl"
dist.init_process_group(
    backend="smddp", ## SageMaker DDP, replacing "nccl"
    timeout=timedelta(minutes=5),  ## default 20 minutes?
)  
## Enables cuDNN's auto-tuner to find the best algorithm for the hardware
torch.backends.cudnn.benchmark = True


## For WebDataset
def get_shard_number(path):
    ## e.g. "train/train-shard-{000000..000001}.tar", 2 shards
    start, _, end = path.split('{')[-1].split('}')[0].split('.')
    if end is None:
        return int(start)
    else:
        return int(end)-int(start)+1

## For WebDataset
class image_transform:
    def __call__(self, x):
        return Image.open(io.BytesIO(x))

## For WebDataset    
train_transform = transforms.Compose([
    image_transform(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
])  

## For WebDataset
def label_transform(x):
    ## Original lables are (1,2,3,4,5)
    ## Convert to (0,1,2,3,4)
    return torch.tensor(int(x.decode())-1, dtype=torch.int64)

## SageMaker DDP
def average_gradients(model):
    for param in model.parameters():
        dist.all_reduce(
            param, 
            op=dist.ReduceOp.SUM, 
            group=dist.group.WORLD, 
            async_op=False)
        param.data /= float(dist.get_world_size())

## For argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('⚠️ Boolean value expected')

## For argparse    
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

## Custom learning rate step
def adjust_learning_rate(current_lr, lr_sched_step_size, epoch):
    """
    Sets the learning rate to the initial LR 
    decayed by 1/10 every lr_sched_step_size epochs
    """
    lr = current_lr * (0.1 ** (epoch//lr_sched_step_size))
    for param_group in task.optimizer.param_groups:
        param_group['lr'] = lr
    return lr

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
    eval_loss_epoch = 0.
    correct = 0.
    num_samples = 0.
    data_loader = task.val_loader if phase=='val' else task.test_loader
    with torch.no_grad():
        for data, target in data_loader:  ## PyTorch loader & WebDataset loader
            num_samples += len(target)
            data, target = data.to(task.config.device), target.to(task.config.device)
            output = task.model(data)
            eval_loss_epoch += task.val_criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    eval_loss_epoch /= num_samples 
    if phase=='val': 
        task.early_stopping(eval_loss_epoch)
        if task.config.wandb:
            wandb.log({f"{phase}_loss_epoch": eval_loss_epoch}, 
                      step=task.step_counter.total_steps)
    accuracy = 100.*correct/num_samples 
    print(
        "👉 {}: Average loss: {:.4f}, Accuracy: {:.0f}/{:.0f} ({:.2f}%)\n".format(
            phase.upper(),
            eval_loss_epoch, 
            correct, 
            num_samples, ## val/test data size
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
    task.model = getattr(torchvision.models, task.config.model_arch)() 
    task.model.fc = nn.Linear(
        task.model.fc.in_features, 
        task.config.num_classes)  # Adjust for the number of classes
    torch.nn.init.kaiming_normal_(task.model.fc.weight)  # Initialize new layers
    task.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(task.model)  ## PyTorch DDP
    ## Wrap model with SMDDP
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
    print(f"👉 Model saved at '{path}'")


def main(task):  ## rank is auto-allocated by DDP when calling mp.spawn
    '''
    Train, eval, test, and save the model
    '''
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
        transforms.Resize((224, 224)),  
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ]) 


    path = f"pipe:aws s3 cp {task.config.train_data_path} -"
    train_dataset = (
        wds.DataPipeline(
            wds.SimpleShardList(path),
            # at this point we have an iterator over all the shards
            wds.shuffle(1000),
            wds.split_by_node,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(),
            # this shuffles the samples in memory
            wds.shuffle(1000),
            # this decodes image and json
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
        wds.DataPipeline(
            wds.SimpleShardList(path),
            wds.tarfile_to_samples(),
            wds.to_tuple('image', 'label'), 
            wds.map_tuple(
                val_transform, 
                label_transform,  
            ),
            wds.batched(task.config.batch_size),
        ) 
    )
    path = f"pipe:aws s3 cp {task.config.test_data_path} -"
    test_dataset = (
        wds.DataPipeline(
            wds.SimpleShardList(path),
            wds.tarfile_to_samples(),
            wds.to_tuple('image', 'label'),
            wds.map_tuple(
                val_transform, 
                label_transform,  
            ),
            wds.batched(task.config.batch_size),
        ) 
    )
    
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
    ## run Val and test on rank 0 node only
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

    ## pre-calculated class weights for streamed data
    if ((task.config.class_weights_dict is None) or (task.config.class_weights_dict=={}) 
        or (task.config.class_weights_dict=='')):
        class_weights = None
    else:
        classes = np.unique(list(task.config.class_weights_dict.keys())) ## It has to be sorted to match the output
        task.config.num_classes = len(classes)  ## get number of total classes for net creation
        class_weights = [task.config.class_weights_dict[k] for k in classes]
        class_weights = torch.tensor(
            class_weights, 
            dtype=torch.float32).to(task.config.device)

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
    task.train_criterion = nn.CrossEntropyLoss(weight=class_weights) # loss per step 
    task.val_criterion = nn.CrossEntropyLoss(
        weight=class_weights, 
        reduction="sum")  ## loss per epoch
    if task.config.debug and task.hook is not None:
        task.hook.register_loss(task.train_criterion)
        task.hook.register_loss(task.val_criterion)
    if task.config.opt_type=='adamw':
        task.optimizer = optim.AdamW(
            task.model.parameters(), 
            lr=task.config.opt_learning_rate * dist.get_world_size(),  ## SMDDP
            weight_decay=task.config.opt_weight_decay,
        )  
        task.scheduler = optim.lr_scheduler.StepLR(
            task.optimizer, 
            step_size=task.config.lr_sched_step_size, 
            gamma=task.config.lr_sched_gamma)  # Reduce LR every 6 epochs by 0.5
    elif task.config.opt_type=='sgd':
        task.optimizer = torch.optim.SGD(
            task.model.parameters(), 
            lr=task.config.opt_learning_rate * dist.get_world_size(),  ## SMDDP
            momentum=task.config.opt_momentum,
            weight_decay=task.config.opt_weight_decay
        )
    '''
    TODO: Call the train function to start training your model
          Remember that you will need to set up a way to get training data from S3
    '''
    # ===========================================================#
    # 5. Pass the SMDebug hook to the train and test functions.  #
    # ===========================================================#
    tensor_early_stop = torch.tensor(0, dtype=torch.int32).to(task.config.device)
    for epoch in range(task.config.epochs):
        task.current_epoch = epoch
        if dist.get_rank()==0:
            print(f"👉 Train Epoch: {epoch}, "
                  f"Learning Rate: {task.optimizer.param_groups[0]['lr']}")
            
        train(task)

        if dist.get_rank()==0:
            eval(task, phase='val')
            ## adjust optimizer learning rate
            if task.config.opt_type=='adamw':
                task.scheduler.step()  
            elif task.config.opt_type=='sgd':
                adjust_learning_rate(
                    task.config.opt_learning_rate, 
                    task.config.lr_sched_step_size,
                    task.current_epoch+1
                )
        if task.early_stopping.early_stop: 
            ## Aggregate the early stop decision across all nodes
            tensor_early_stop = torch.tensor(1, 
                dtype=torch.int32).to(task.config.device)
            dist.all_reduce(tensor_early_stop, op=dist.ReduceOp.MAX)  ## SUM could get large
            print(f"⚠️ Early stopping all-reducing "
                    f"{tensor_early_stop.item()} to Rank {dist.get_rank()}...")
        if tensor_early_stop.item()!=0:
            print(f"⚠️ Early stopping at epoch {task.current_epoch} "
                  f"on Rank {dist.get_rank()}")
            break

    if dist.get_rank()==0:
        print("🟢 Start testing...")
        eval(task, phase='test')
        print("🟢 Start saving the trained model...")
        save(task)


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

    parser=argparse.ArgumentParser()
    ## Hyperparameters passed by the SageMaker estimator
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--opt-type', type=str, default='adamw')
    parser.add_argument('--opt-learning-rate', type=float, default=1e-4)
    parser.add_argument('--opt-weight-decay', type=float, default=1e-4) 
    parser.add_argument('--opt-momentum', type=float, default=0.9) 
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
    parser.add_argument('--num-classes', type=int, default=0)
    parser.add_argument('--class-weights-dict', type=str2dict, default={})
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

    main(task)

    ## Finish wandb run
    if task.config.wandb and dist.get_rank()==0:
        try:
            wandb.config.update(task.config.__dict__, 
                                allow_val_change=True)
        except Exception as e:
            print(f"⚠️ Updating wandb config failed: {e}")
        wandb.finish()