## https://sagemaker.readthedocs.io/en/stable/api/training/sdp_versions/v1.0.0/smd_data_parallel_pytorch.html

## TODO: Import your dependencies.
## For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
# from torchvision import datasets, transforms
from torchvision import transforms
import webdataset as wds
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import os
import argparse
import wandb

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow truncated images

## Move the import to the main check part of this script
## TODO: Import dependencies for Debugging andd Profiling
# ====================================#
# 1. Import SMDebug framework class.  #
# ====================================#
# import smdebug.pytorch as smd

## PyTorch Distributed Data Parallel (DDP) 
# import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group
## SageMaker distributed data parallel library PyTorch API
import smdistributed.dataparallel.torch.distributed as dist
from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
dist.init_process_group(backend="smddp")  ## SageMaker DDP, replacing "nccl"
# Pin GPU to be used to process local rank (one GPU per process)
torch.cuda.set_device(dist.get_local_rank())



## for WebDataset
def identity(x):
     return x



## SageMaker DDP
def average_gradients(model):
    for param in model.parameters:
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
    if dist.get_rank()==0:
        print(f"👉 Train Epoch: {task.current_epoch}")
    for batch_idx, (data, target) in enumerate(task.train_loader):
        task.step_counter()
        data, target = data.to(task.config.device), target.to(task.config.device)  ## inputs, labels
        task.optimizer.zero_grad()
        output = task.model(data)
        loss = task.train_criterion(output, target)
        if task.config.wandb and dist.get_rank()==0:
            wandb.log({"train_loss": loss.item()}, step=task.step_counter.total_steps)
        loss.backward()
        average_gradients(task.model)  ## SageMaker DDP
        task.optimizer.step()
        if batch_idx%100==0 and dist.get_rank()==0:  ## Print every 100 batches
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f}".format(
                    task.current_epoch,
                    batch_idx * len(data),
                    len(task.train_loader.dataset),
                    100.0 * batch_idx / len(task.train_loader),
                    loss.item(),
                )
            )



def eval(task, phase='eval'):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    # ===================================================#
    # 3. Set the SMDebug hook for the validation phase. #
    # ===================================================#
    if task.config.debug and task.hook: 
        task.hook.set_mode(smd.modes.EVAL)
    task.model.eval()
    test_loss = 0.
    correct = 0.
    data_loader = task.val_loader if phase=='eval' else task.test_loader
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(task.config.device), target.to(task.config.device)
            output = task.model(data)
            test_loss += task.val_criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(data_loader.dataset)
    if phase=='eval': 
        task.early_stopping(test_loss)
        if task.config.wandb and dist.get_rank()==0:
            wandb.log({f"{phase}_loss_epoch": test_loss}, step=task.step_counter.total_steps)
    accuracy = 100.*correct/len(data_loader.dataset)
    print(
        "\n👉 {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            phase.upper(),
            test_loss, 
            correct, 
            len(data_loader.dataset), 
            accuracy
        )
    )
    if phase=='eval' and task.config.wandb:
        wandb.log(
            {f"rank {dist.get_rank()}: {phase}_accuracy_epoch": accuracy}, 
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
    task.model.to(task.config.device)
    # task.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(task.model)  ## PyTorch DDP
    ## Wrap model with smdistributed.dataparallel's DistributedDataParallel
    task.model = DDP(task.model, 
                     device_ids=[torch.cuda.current_device()])  ## single-device Torch module  
    



def save(task):
    '''
    Save the model to the model_dir
    '''
    task.model.eval()
    path = os.path.join(task.config.model_dir, 'model.pth')
    ## save model weights only
    with open(path, 'wb') as f:
        torch.save(task.model.state_dict(), f)
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
    print(f"Model saved at '{path}'")



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
    print(f"👉 Device: {task.config.device}")
    print(f"   Device rank: {dist.get_rank()}")  ## SMDDP
    print(f"   Device local rank: {dist.get_local_rank()}")  ## SMDDP
    task.config.num_classes = len(train_dataset.classes)
    task.config.num_cpu = os.cpu_count()  ## for data loaders

    ## before initializing the group process, call set_device, 
    ## which sets the default GPU for each process. This is important 
    ## to prevent hangs or excessive memory utilization on GPU:0.
    # ddp_setup(rank, task.config.world_size)  ## DDP

    train_transform = transforms.Compose([
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
        transforms.Resize((224, 224)),  ## default: interpolation=InterpolationMode.BILINEAR
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ]) 

    ## Replace TorchVision.datasert.ImageFolder() with WebDataset.dataset() pipe
    # train_dataset = datasets.ImageFolder(task.config.train, transform=train_transform)
    # val_dataset = datasets.ImageFolder(task.config.validation, transform=val_transform)
    # test_dataset = datasets.ImageFolder(task.config.test, transform=val_transform)

    ## For data distributed training, use torch.utils.data.DistributedSampler or WebDataset? 
    path = f"pipe:aws s3 cp {task.config.train} -"
    train_dataset = (
        wds.WebDataset(path, 
                       shardshuffle=True, ## Shuffle shards
                       nodesplitter=wds.split_by_worker)  ## distributed training
            .shuffle(1000)  # Shuffle dataset
            .decode("pil")  
            .to_tuple("jpg", "cls")  # Tuple of image and label; specify file extensions
            .map_tuple(train_transform, identity)  # Apply the train transforms to the image
    )
    path = f"pipe:aws s3 cp {task.config.validation} -"
    val_dataset = (
        wds.WebDataset(path, 
                       shardshuffle=False,  ## Shuffle shards
                       nodesplitter=wds.split_by_worker) ## distributed
            .decode("pil")  
            .to_tuple("jpg", "cls")  # Tuple of image and label; specify file extensions
            .map_tuple(val_transform, identity)  # Apply the train transforms to the image
    )
    path = f"pipe:aws s3 cp {task.config.test} -"
    test_dataset = (
        wds.WebDataset(path, 
                       shardshuffle=False,  ## Shuffle shards
                       nodesplitter=wds.split_by_worker)  ## distributed 
            .decode("pil")  
            .to_tuple("jpg", "cls")  # Tuple of image and label; specify file extensions
            .map_tuple(val_transform, identity)  # Apply the train transforms to the image
    )

    ## handle class imbalance. class weights will be used in the loss functions.
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(train_dataset.targets), 
        y=train_dataset.targets)
    class_weights = torch.tensor(
        class_weights, 
        dtype=torch.float32).to(task.config.device)
    
    ## SMDDP: set num_replicas and rank in Torch DistributedSampler
    train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank())
    val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank())
    test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank())
    task.train_loader = DataLoader(
        train_dataset, 
        batch_size=task.config.batch_size, 
        shuffle=False,  ## Don't shuffle for Distributed Data Parallel (DDP)  
        sampler=train_sampler, # Use the Distributed Sampler
        num_workers=task.config.num_cpu,
        pin_memory=True)
    task.val_loader = DataLoader(
        val_dataset, 
        batch_size=task.config.batch_size, 
        shuffle=False,   ## Don't shuffle for eval anyway
        sampler=val_sampler, # Use the Distributed Sampler
        num_workers=task.config.num_cpu,
        pin_memory=True)
    task.test_loader = DataLoader(
        test_dataset, 
        batch_size=task.config.batch_size, 
        shuffle=False,  ## Don't shuffle for eval anyway
        sampler=test_sampler, # Use the Distributed Sampler
        num_workers=task.config.num_cpu,
        pin_memory=True)
    

    ## TODO: Initialize a model by calling the net function
    create_net(task)

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
    if task.config.debug and task.hook:
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
        task.train_loader.sampler.set_epoch(epoch)  ## for DDP
        train(task)
        eval(task, phase='eval')
        if task.early_stopping.early_stop:
            print("⚠️ Early stopping")
            break
        task.scheduler.step()  ## Update learning rate after every epoch
        print(f"👉 Train Epoch: {epoch+1}, Learning rate: {task.optimizer.param_groups[0]['lr']}")

    ## TODO: Test the model to see its accuracy
    print("🟢 Start testing...")
    eval(task, phase='test')
    # destroy_process_group()  ## PyTorch DDP

    ## TODO: Save the trained model
    if dist.get_rank()==0:  ## DDP only save one
        save(task)



if __name__=='__main__':



    ## SageMaker DDP
    is_initialized = dist.is_initialized()
    if is_initialized:
        print("🟢 SageMkaer DDP is initialized.")
    else:
        raise RuntimeError("⚠️ SageMaker DDP is not initialized.")
    print(f"👉 Total GPU count: {dist.get_world_size(group=dist.group.WORLD)}")
    print(f"👉 Rank: {dist.get_rank(group=dist.group.WORLD)}")  ## the rank of the worker node
    print(f"👉 Local Rank: {dist.get_local_rank()}") ## the rank of the GPU

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
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
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
    print(f"👉 configs: {task.config.__dict__}")

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