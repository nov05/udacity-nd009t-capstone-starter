# 🟢 **Capstone Project: Teratype-Scale Machine Learning with AWS**    
**Udacity AWS Machine Learning Engineer Nanodegree (ND189)**   
Amazon Bin Object Counting, a demonstration of end-to-end machine learning engineering skills on AWS  

### 🏷️ **Technical highlights:** 

All the techniques listed below can be seamlessly applied to **large-scale datasets**, including terabyte-scale training data, ensuring efficient processing and scalability without overwhelming infrastructure resources. Together, they form a comprehensive machine learning development workflow—covering **ETL**, **EDA**, **data ingestion**, **training** (with tuning, debugging, and profiling), and **inference**. While **monitoring** was not implemented in this project, it has been demonstrated in previous hands-on exercises of this course.

- For demonstration purposes, a subset of **10,441 samples** with **5 classes** was selected from the original dataset of over **500,000+ entries**.
  
- **Exploratory Data Analysis (EDA)** was performed using **AWS Athena** CTAS and **Trino SQL** queries on the 10K metadata JSON files. The **10,441 JSON files** were efficiently consolidated into **21 SNAPPY-compressed Parquet files** in just **3.6 seconds**.
  
- **AWS Glue-Spark** jobs were developed locally using **Docker** and **VS Code** for further EDA tasks, such as generating visualizations of the class distribution and other data insights.
  
- To prepare the dataset for training, **AWS SageMaker's ScriptProcessor** was utilized in combination with a custom **Docker** image uploaded to **AWS ECR**. The **10K dataset** was successfully converted into **WebDataset** .tar files for streamlined data loading during training.

- **WebDataset** (a subclass of PyTorch IterableDataset) is leveraged with the `'pipe'` command to stream data directly from **S3** to the **SageMaker** training instance(s). This approach enables efficient handling of **terabyte-scale datasets** without needing to copy the entire dataset to the training instance(s) at once. As a result, there’s no need for instances with large storage or external mounts like **EFS**, significantly reducing infrastructure costs. Additionally, this method offers a cost-effective alternative to using **Amazon FSx**, as it only incurs a fraction of the cost while still enabling large-scale data processing.  

  * `FastFile` input mode streams data in real-time and is cost-effective. However, when streaming smaller files directly from Amazon S3, the overhead of dispatching a new GET request for each file becomes significant relative to the file transfer time (even with a highly parallel data loader and prefetch buffer). This results in lower overall throughput for `FastFile Mode`, causing an I/O bottleneck during training.

  * In reality, the **Amazon Bin Image Dataset** consists of 50K sample file pairs, with a total size of around 60GB—falling comfortably within the 50-100GB range—so we can use `File Mode`. However, for the purpose of this project, we are simulating tens of terabytes of data.   

- **AWS SageMaker Distributed Data Parallel (SMDDP)** framework is combined with **WebDataset** for distributed training. SMDDP efficiently manages tasks such as model replication across GPU nodes, asynchronous training, and synchronization of model weights across nodes. Meanwhile, **WebDataset** handles the shuffling, transforming, node-wise data splitting, and batching of training data, ensuring seamless data distribution for each node during training.  

- **Technical tips:**  
  * The `WebDataset` class inherits from PyTorch's `IterableDataset`, which isn't compatible with the standard PyTorch `DataLoader`. Instead, use the `WebLoader` from WebDataset to create and iterate batches of streamed data.  
  * To implement **early stopping** for distributed training, for example, when validation loss doesn't improve for 5 epochs, use **SMDDP**'s [`dist.all_reduce(tensor_early_stop, op=dist.ReduceOp.MAX)`](https://pytorch.org/tutorials/intermediate/dist_tuto.html) to notify all nodes to stop.  
  * To address **class imbalance** in large streamed training datasets, we can leverage big data analytics tools during pre-processing to calculate class weights, which can then be passed to the training instance as a hyperparameter. 
  * **Wandb** (Weights and Biases) is used for experiment tracking and visualizing machine learning training runs, which allows us to log training information, such as loss, accuracy, and other metrics, as well as visualize training curves in real-time.
  * Check the pricing and **AWS Service Quotas** at the account level when selecting SageMaker EC2 instances. By default, there is no quota for GPU **spot instances** available.  
  * You can run **SageMaker** locally using your preferred IDE, which helps you avoid the costs associated with SageMaker Studio. Just configure your AWS config and credentials files on your local machine, ensure that your SageMaker session starts with the correct profile, and retrieve the appropriate execution role.
     


<br><br><br>  

---  

# 👉 **Project Submission**

### 🏷️ **Environment and Services**

* [Local conda env](https://gist.github.com/nov05/a6eccfd88ef180d5cae0d0d0e2fc646d?permalink_comment_id=5425643#gistcomment-5425643)  
* Windows 11 (OS), VS Cdoe (IDE), AWS SageMaker / Athena / S3 / ECR / IAM / CloudWatch, Wandb, Docker
* Folder structure:
  ```
    /starter
      ├── # ETL, EDA, training/tuning/deployment/etc. notebooks
      └── `AWS Athena Trino SQL.md`  # Athena CTAS and queries

    /docker_workspace   # folder attached to the local Glue-Spark Docker container
      └── `aft-vbi-pds.ipynb`  # PySpark analysis of the metadata

    /scripts_train
      └── `train_v1.py`  # Major training script for SageMaker Torch estimator

    /scripts_process
      └── `convert_to_webdataset_10k.py`  # Major preprocessing script for SageMaker Processor

    /docker_process
      └── `dockerfile`  # To create a custom image for SageMaker Processor

    /scripts_inference
      └── # deployment scripts

    /examples
      └── # experiments

    /data   # in the `.gitignore` file
      └── # local data

    /secrets   # in the `.gitignore` file
      └── # AWS account numbers, profile names, etc.; wandb secret
  ```


### 🏷️ **Metadata Exploratory Data Analysis (EDA)**

* Check [the PySpark EDA notebook](https://github.com/nov05/udacity-nd009t-capstone-starter/blob/master/docker_workspace/aft-vbi-pds.ipynb) on a subset of the metadata (10K out of 500K JSON files from the original dataset)

  * Demo video: [Query and consolidate a large number of small JSON files with AWS Athena <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/icons/youtube_red_2000x2000.png" width=20>](https://www.youtube.com/watch?v=DMazwtXpF8I)  
    Check the [AWS Athena Trino SQL](https://github.com/nov05/udacity-nd009t-capstone-starter/blob/master/starter/AWS%20Athena%20Trino%20SQL.md) used here

  * Demo video: [Develop local AWS Glue Spark jobs with Docker and VS Code <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/icons/youtube_red_2000x2000.png" width=20>](https://www.youtube.com/watch?v=Kqw_1q9O2NQ)  
    Check [the Text format tutorial](https://docs.google.com/document/d/1FtVdxZ283kILxVvl02-FmvLilk3uemvU_vIaJct2p5w)   

  <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/metadata_01_expected%20quantity.png" width=300> <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/metadata_02_item%20quantity.png" width=300> <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/metadata_03_hlw_weight.png" width=365>   
 
* From the target (expected quantity) distribution plot, we can see that there is class imbalance. Additionally, the item quantity distribution plot shows that most items only have one or two images.  

### 🏷️ **Data Preparation**  

* Shuffle the dataset, then split it into train, validation, and test sets. Use WebDataset to convert pairs of `.jpg` and `.json` files into `.tar` files, where each file contains a shard with (\_\_key\_\_, jpg, class) data — e.g., a shard size of 1,000 samples. Ultimately, the 10,441 samples are saved as 7, 2, and 2 `.tar` files for the train, validation, and test sets, respectively, and stored in an S3 bucket with the prefix `s3://p5-amazon-bin-images/webdataset/train/`, `~/val/` and `~/test/`.

  * ✅ Check [the docker file](https://github.com/nov05/udacity-nd009t-capstone-starter/blob/master/docker_process/dockerfile) to build the custom image for **SageMaker Processor**  

  * ✅✅ Check [the processing notebook](https://github.com/nov05/udacity-nd009t-capstone-starter/blob/master/starter/ETL.ipynb) and [script](https://github.com/nov05/udacity-nd009t-capstone-starter/blob/master/scripts_process/convert_to_webdataset_10k.py)  

    * In the processing script  
      ```python
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
      ```

  <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/2025-02-08%2011_12_08-p5-amazon-bin-images%20-%20S3%20bucket%20_%20S3%20_%20us-east-1.jpg" width=800>  

* During training, we can stream the data from S3 to the training instances by passing the dataset paths as hyperparameters. In the training script, we use **WebDataset** `DataPipeline` and `WebLoader` to stream, buffle, shuffle, split by node (GPUs), transform, batch the data.

  ```python
  data_base_path = "s3://p5-amazon-bin-images/webdataset/"
  train_data_path = data_base_path + "train/train-shard-{000000..000007}.tar"
  ```
  ```python
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
        wds.to_tuple('image', 'label'),
        wds.map_tuple(
            train_transform, 
            label_transform,  
        ),
        wds.shuffle(1000),
        wds.batched(task.config.batch_size),
      )
  ) 
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
  ```

### 🏷️ **Distributed Training and Hyperparameters Tuning (HPO)**  

* Use **SageMaker Distributed Data Parallel (SMDDP)** framework for distributed training  

  * ✅✅✅ Check [the SageMaker notebook](https://github.com/nov05/udacity-nd009t-capstone-starter/blob/master/starter/01_sagemaker_10k_adamw.ipynb) and [the training script](https://github.com/nov05/udacity-nd009t-capstone-starter/blob/master/scripts_train/train_v1.py)   

    * Beginning of the script
    ```python
    import smdistributed.dataparallel.torch.distributed as dist
    from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
    dist.init_process_group(
        backend="smddp", ## SageMaker DDP, replacing "nccl"
        timeout=timedelta(minutes=5),  ## default 20 minutes?
    )  
    ```

    * When creating the neural network  
    ```python
    def create_net(task):
        ...
        task.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(task.model)  ## PyTorch DDP
        ## Wrap model with SMDDP
        task.model = DDP(task.model.to(task.config.device), 
                        device_ids=[torch.cuda.current_device()])  ## single-device Torch module 
    ```

    * In the main function  
    ```python
    create_net(task)
    ## SMDDP: Pin each GPU to a single distributed data parallel library process.
    torch.cuda.set_device(dist.get_local_rank())
    task.model.cuda(dist.get_local_rank())
    ```

    * Test and save model on one node only
    ```python
    if dist.get_rank()==0:
        print("🟢 Start testing...")
        eval(task, phase='test')
        print("🟢 Start saving the trained model...")
        save(task)
    ```

* Use WebDataset to convert and stream datasets to the training instances from S3   

* Use 2 `ml.g4dn.xlarge` GPU instances for training and HPO  

  AWS SageMaker HPO Warm Pools improve hyperparameter optimization efficiency, especially in **distributed training**, by reusing instances between jobs, reducing provisioning time, lowering latency, and optimizing resource use. This is particularly valuable for distributed training, where large instance clusters are reused, speeding up experimentation and cutting costs while enabling faster convergence on optimal hyperparameters.
  <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/2025-02-08%2007_19_39-Amazon%20SageMaker%20AI%20_%20us-east-1.jpg" width=800>  

* Use **wandb** to visulize the training curves. 

  <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/2025-02-08%2011_24_28-udacity-awsmle-resnet34-amazon-bin%20Workspace%20%E2%80%93%20Weights%20%26%20Biases.jpg" width=800>  

* Check **AWS CloudWatch** logs

  <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/2025-02-08%2011_37_02-Amazon%20SageMaker%20AI%20_%20us-east-1.jpg" width=800>  

  <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/2025-02-08%2013_15_21-CloudWatch%20_%20us-east-1.jpg" width=800>  
 
  Take note of the MPI logs. According to [the official documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel-config.html), the SageMaker Distributed Data Parallel library uses the **Message Passing Interface (MPI)**, a widely used standard for communication between nodes in high-performance clusters, along with NVIDIA’s **NCCL library** for GPU-level communication.

  * In the training notebook:  
    ```python
    from sagemaker.pytorch import PyTorch
    estimator = PyTorch(
        ...
        distribution={"smdistributed": {"dataparallel": { "enabled": True}}},
    )
    ```

* Find the best hyperparameter job from HPO. You can deploy from the job name, or from the model S3 URI. 

  <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/2025-02-08%2011_34_35-01_sagemaker_10k_adamw.ipynb%20-%20udacity-nd189-aws-mle-nanodegree%20(Workspace)%20-%20Vi.jpg" width=600>

  **Note:** 

  1. Model performance and training accuracy are not the focus of this project, as outlined in the project proposal. Instead, the objective is to build a machine learning training pipeline for handling relatively large datasets in AWS.

  2. Since I focused primarily on **mocking big data ETL** and **distributed training** in this project, and have already gained experience with debugging and profiling (as well as wandb sweeping, deployment, inference, etc.) in previous projects, I'll be skipping those steps here. (Check [the notebooks from P3 dog breed image classification](https://github.com/nov05/udacity-CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter?tab=readme-ov-file).)  

  3. That said, I’d like to mention the training results using 10K out of 500K samples. The best accuracy achieved was **around 36%**, compared to 55.67% achieved by others using the full 500K dataset. My speculation is that this is likely close to the best possible result from such a small dataset (1/50th of the original). Given that this is a demo project with limited resources, I’ll leave the results as they are.

  4. Please refer to the source code notebooks, scripts, and project notes for much more technical details. Here, I’m only providing a high-level overview of the work, as including everything would make the `README.md` too lengthy. Lol.



### 🏷️ **Notes**

* GitHub Issues (closed): [How to Implement Early Stopping with WebDataset in SageMaker Distributed Data Parallel (SMDDP) Framework? #446](https://github.com/webdataset/webdataset/issues/446)  
* [SageMaker input mode and WebDataset](https://docs.google.com/document/d/1SNQuYrCOy6s5Zg3NXdkrujOyDPP1VA4b8HKAqEoJvM4)   
* [SageMaker PyTorch distributed training](https://docs.google.com/document/d/12yN589I95IdyJjOwoxH5uQf08bfCvnXvIYRYChFw8R8)       
* [Different Levels of AWS Resources for Machine Learning Model Training and Deployment](https://gist.github.com/nov05/6f39c83c143d91175075fb8e7e871d0c)    
* [Tutorial:](https://docs.google.com/document/d/17KzWVf84xQJVNH1jd6yh_FLgr781QcdKng1JIF6P5X4) Create custom docker image for SageMaker data processing jobs, create AWS ECR private repo, and upload the image to the repo   

✅ [All other notes for the nanodegree](https://drive.google.com/drive/folders/1-BRvqMlMbk1E6kV6BALTLhXRyqJMWPNE) on Google Drive  

<br><br><br>

---  

# 👉 **Project Proposal** 

### **Domain Background**  

  * In distribution centers, bins often carry multiple objects, making accurate counting important for inventory and shipments. This project focuses on building a model that can **count the number of objects in a bin from a photo**, helping to track inventory and ensure correct deliveries.  

  * Machine learning in image processing has evolved significantly over the past few decades. Early approaches in the 1960s and 1970s relied on basic statistical methods and hand-crafted features to analyze images. In the 1990s, **support vector machines (SVMs)** and **k-nearest neighbors (k-NN)** became popular for tasks like image classification.

    The real revolution came with the rise of deep learning in the 2010s, particularly through **convolutional neural networks (CNNs)**, pioneered by Yann LeCun in the late 1990s. **AlexNet**'s win in the ImageNet competition in 2012 demonstrated the power of deep CNNs for large-scale image recognition, leading to a boom in deep learning research. This was followed by more advanced models like **VGG**, **ResNet**, and **EfficientNet**.

    Today, transfer learning and pre-trained models are commonly used to tackle image processing tasks, significantly reducing the time and resources needed for training. Emerging techniques like **GANs (Generative Adversarial Networks)** are now being used for image generation and manipulation. The field continues to advance with innovations in architectures, optimization, and real-time image processing.

### **Problem Statement**  

  * The dataset is huge and the training might take long time.
  * The images are blurry, in different sizes, with noises such as tapes over the bin. Objects in the bin are different products in all kinds of shapes and size, and might overlap each other. All these increase the difficulty in prediction.  
    <img src="https://raw.githubusercontent.com/silverbottlep/abid_challenge/refs/heads/master/figs/abid_images.png" width=600>  
  * The focus of this project is on building a machine learning training pipeline in AWS, with **model performance not being a concern**.  

### **Solution Statement**   

  * For this project, I'll use **ResNet34** and train it from scratch, likely for only a few epochs, since accuracy isn't the goal.   
  * Leverage AWS SageMaker features like Pipe Mode, distributed training, and hyperparameter tuning for faster training. Additionally, use Spot Instances to optimize costs if possible.

### **Datasets and Inputs**

  * I will use the `Amazon Bin Image Dataset`, which contains **535,234 images** of bins holding one or more objects. These objects include **459,476 different products** in various shapes and sizes. Each image is accompanied by a metadata file that provides details like the number of objects, image dimensions, and object type. Our task is to classify the number of objects in each bin.  
  * For this project, I will only use images of bins containing fewer than 6 objects (**0–5 objects**, corresponding to **6 classes**).  

### **Benchmark Models**  

  * Random class baseline (accuracy): 20.08%
  * Largest class baseline (accuracy): 22.27%
  * [**ResNet34** + SGDR (accuracy): 53.8%](https://github.com/pablo-tech/Image-Inventory-Reconciliation-with-SVM-and-CNN/tree/master) (2018, Pablo Rodriguez Bertorello, Sravan Sripada, Nutchapol Dendumrongsup)    
    This team also stated that they were able to improve model accuracy by 80% using a CNN compared to an **SVM**, achieving an overall 324% improvement over random.  
  * [**ResNet34** + SGD (accuracy): **55.67%**](https://github.com/silverbottlep/abid_challenge) (2018, Eunbyung Park)  

    | Accuracy (%)| RMSE (Root Mean Square Error) |
    |-------------|-------------------------------|
    | 55.67       | 0.930                         |

    | Quantity | Per class accuracy(%) | Per class RMSE |
    |----------|-----------------------|----------------|
    | 0        | 97.7                  | 0.187          |
    | 1        | 83.4                  | 0.542          |
    | 2        | 67.2                  | 0.710          |
    | 3        | 54.9                  | 0.867          |
    | 4        | 42.6                  | 1.025          |
    | 5        | 44.9                  | 1.311          |

### **Evaluation Metrics**

  * **Accuracy**: whether predicted object numbers matches the actual numbers
  * **RMSE (Root Mean Squared Error)**: Indicates how close the predicted object numbers are to the actual values, with larger errors being penalized more.  
  * If the model begins training and shows **steady improvements in accuracy** with each epoch, we can confirm that the training pipeline is functioning properly, which meets the project's goal. There's no need to train until the highest accuracy is achieved, as accuracy isn't the concern for this project.  

### **Project Design**

  * Exploratory Data Analysis (check [the notebook](https://github.com/nov05/udacity-nd009t-capstone-starter/blob/master/starter/EDA.ipynb))  
  * Limit the project to only use images of bins containing fewer than 6 objects (**0~5 objects, 6 classes**). 
  * There are over 500,000 images in the dataset. After sampling the image sizes, I found that they range from 40 to 120 KB, while the JSON files range from 1 to 3 KB each. This means the total size of the image data is between 20 and 60 GB, and the JSON data is between 0.5 and 1.5 GB. Hence we choose **fast file mode**, or **pipe mode** as the training data input mode.  
    https://docs.aws.amazon.com/sagemaker/latest/dg/model-access-training-data.html
    <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/2025-01-09%2011_13_55-Setting%20up%20training%20jobs%20to%20access%20datasets%20-%20Amazon%20SageMaker%20AI.jpg" width=600>   
  * The original dataset actually fits **Scenario C**: your dataset is too large for File mode, or has many small files (which you can’t serialize easily), or you have a random read access pattern. Hence, **FSx for Lustre** would be a great choice. However, the estimated throughput cost would be around $300 for 60GB of data, so I’ve decided not to use it as input mode.    
    https://aws.amazon.com/blogs/machine-learning/choose-the-best-data-source-for-your-amazon-sagemaker-training-job/      
    <img src="https://raw.githubusercontent.com/nov05/pictures/refs/heads/master/Udacity/20241119_aws-mle-nanodegree/ML-2979-image005.jpg" width=400>  
  * To prototype the training data input process, we can use the 10,441 images listed in the `file_list.json` from the starter repository.  
  * Use SageMaker script mode with an AWS GPU instance like `g4dn.xlarge` and enable multi-instance training.  
  * Hyperparameters tuning
  * Deploy endpoint  
  * Testing the endpoint
  * Clean up resources


<br><br><br>  

---  

# 👉 **Project Overview: Inventory Monitoring at Distribution Centers**  

Distribution centers often use robots to move objects as a part of their operations. Objects are carried in bins which can contain multiple objects. In this project, you will have to build a model that can count the number of objects in each bin. A system like this can be used to track inventory and make sure that delivery consignments have the correct number of items.

To build this project you will use AWS SageMaker and good machine learning engineering practices to fetch data from a database, preprocess it, and then train a machine learning model. This project will serve as a demonstration of end-to-end machine learning engineering skills that you have learned as a part of this nanodegree.

## 🏷️ **How it Works**

To complete this project we will be using the <a href="https://registry.opendata.aws/amazon-bin-imagery/" target="_blank">Amazon Bin Image Dataset</a>. The dataset contains 500,000 images of bins containing one or more objects. For each image there is a metadata file containing information about the image like the number of objects, it's dimension and the type of object. For this task, we will try to classify the number of objects in each bin.

To perform the classification you can use a model type and architecture of your choice. For instance you could use a pre-trained convolutional neural network, or you could create your own neural network architecture. However, you will need to train your model using SageMaker.

Once you have trained your model you can attempt some of the Standout Suggestion to get the extra practice and to turn your project into a portfolio piece.

## 🏷️ **Pipeline**

To finish this project, you will have to perform the following tasks:

1. Upload Training Data: First you will have to upload the training data to an S3 bucket.
1. Model Training Script: Once you have done that, you will have to write a script to train a model on that dataset.
1. Train in SageMaker: Finally, you will have to use SageMaker to run that training script and train your model

Here are the tasks you have to do in more detail:

### Setup AWS
To build this project, you wlll have to use AWS through your classroom. Below are your main steps:
- Open AWS through the classroom on the left panel (**Open AWS Gateway**)
- Open SageMaker Studio and create a folder for your project

### Download the Starter Files
We have provided a project template and some helpful starter files for this project. You can clone the Github Repo.
- Clone of download starter files from Github
- Upload starter files to your workspace

### Preparing Data
To build this project you will have to use the [Amazon Bin Images Dataset](https://registry.opendata.aws/amazon-bin-imagery/)
- Download the dataset: Since this is a large dataset, you have been provided with some code to download a small subset of that data. You are encouraged to use this subset to prevent any excess SageMaker credit usage.
- Preprocess and clean the files (if needed)
- Upload them to an S3 bucket so that SageMaker can use them for training
- OPTIONAL: Verify that the data has been uploaded correctly to the right bucket using the AWS S3 CLI or the S3 UI

### Starter Code
Familiarize yourself with the following starter code
- `sagemaker.ipynb`
- `train.py`

### Create a Training Script
Complete the TODO's in the `train.py` script
- Read and Preprocess data: Before training your model, you will need to read, load and preprocess your training, testing and validation data
- Train your Model: You can choose any model type or architecture for this project

### Train using SageMaker
Complete the TODO's in the `sagemaker.ipynb` notebook
- Install necessary dependencies
- Setup the training estimator
- Submit the job

### Final Steps
An important part of your project is creating a `README` file that describes the project, explains how to set up and run the code, and describes your results. We've included a template in the starter files (that you downloaded earlier), with `TODOs` for each of the things you should include.
- Complete the `README` file

## 🏷️ Standout Suggestions

Standout suggestions are some recommendations to help you take your project further and turn it into a nice portfolio piece. If you have been having a good time working on this project and want some additional practice, then we recommend that you try them. However, these suggestions are all optional and you can skip any (or all) of them and submit the project in the next page.

Here are some of suggestions to improve your project:

* **Model Deployment:** Once you have trained your model, can you deploy your model to a SageMaker endpoint and then query it with an image to get a prediction?
* **Hyperparameter Tuning**: To improve the performance of your model, can you use SageMaker’s Hyperparameter Tuning to search through a hyperparameter space and get the value of the best hyperparameters?
* **Reduce Costs:** To reduce the cost of your machine learning engineering pipeline, can you do a cost analysis and use spot instances to train your model?
* **Multi-Instance Training:** Can you train the same model, but this time distribute your training workload across multiple instances?

Once you have completed the standout suggestions, make sure that you explain what you did and how you did it in the `README`. This way the reviewers will look out for it and can give you helpful tips and suggestions!


<br><br><br>  

---  

### **Logs**

2025-01-21 tutorial: How to run local Glue Spark jobs with Docker and VS Code   
2025-01-17 tutorial: Query and consolidate a large number of small JSON files with AWS Athena  
2025-01-09 project proposal    