## **The Foot Ulcer Segmentation Challenge (FUSeg)**  
![Dataset_Image](https://raw.githubusercontent.com/Pele324/ChronicWoundSeg/master/figures/Dataset.png)

FUSeg is a MICCAI endorsed challenge.   
For more details please check our [homepage](https://fusc.grand-challenge.org/FUSeg-2021/) on grand-challenge.org and our [challenge design](https://github.com/uwm-bigdata/wound-segmentation/blob/master/data/Foot%20Ulcer%20Segmentation%20Challenge/FootUlcerSegmentationChallenge2021.pdf).   

# Submission Instructions  
__The brief version__:  
Please submit your algorithm in a ready-to-use Docker container that should be able to generate segmentation masks from the provided test images.

__The detailed version__:  
To make sure that we will be able to run all submitted algorithms, we would like to have your runnable prediction code in a Docker container. There are plenty of online tutorials on how to containerize your algorithm if you are not familiar with the process. We also recommand [Best practices for writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/) to write your Dockerfile for building the container. We will not provide any sample containers, but rather give you the freedom to be in your own familiar environment. To smooth the evaluation process, we kindly ask you to prepare your container in accordance with the requirements listed below:  
1. Provide a brief description of your method in a few sentences.  
2. Let us know the instructions/conmmands to execute in your container to generate segmentation masks. (for example, "put the test images in ./data/test/images/ and run ```python ./predict.py```") Of course it would be better if you already have the test images in the container.
3. If your algorithm require more than 24GB of GPU memory, please let us know and provide info like expected RAM and GPU memory used, expected runtime on CPU/GPU, or anything else we should know before runing it...  
4. Please make sure that we will have all dependencies when we run your container. The best practice is to include the installations of dependencies (```RUN pip install -r requirements.txt```) in the Dockerfile when you build your container.  
6. Please tag your container with "FUSeg2021" followed by your name, organization, or any other reasonable info (for exampple, "FUSeg2021_ChuanboWang_UW-Milwaukee"). 
7. It would be very helpfull to us if your algorithm generates masks as binary images (where pixel value 0 indicates non-wound and pixel value 1 indicates wound) with the same filenames as the original images.
8. Please double check that your cintainer is ready to be run before submission.  
9. Please send the submission docker container to chuanbo@uwm.edu. If your container is too big, upload it to Docker Hub or any cloud storage and send a link to the provided email.

# Evaluation Process
1. We will run your algorithm in your container to generate segmantion masks.  
2. The same evaluation algorithm will be used for all submissions to compare the generated masks with the ground truth masks.  
3. The evaluation algorithm will compute the precision, the average dice similarity coefficient (DSC), and the average intersection over union rate(IOU).  
4. Challenge rankings will be based on average DSC. Precision will be used in case of DSC ties.

# Folder structure
.  
│── train &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; # training dataset  
│   ├── images &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; # training images  
│   └── labels &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; # ground truth masks  
│── validation &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; # validation dataset  
│   ├── images &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; # validation images  
│   └── labels &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; # ground truth masks  
└── test &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; # testing dataset

# Updates  
Mar 9 2021  The training and validation dataset is published.  
Testing data will be released in July 2021.  
Submission deadline update: July 15 2021.  (The deadline was moved forward from Aug 1 2021 to July 15 2021)  
July 5 2021  Additional 200 labeled images are added to the training dataset.  
July 10 2021 A new testing dataset is published for sanity checks and your docker submission.  
July 11 2021 The submission instructions are published.

# Publication
Wang, C., Anisuzzaman, D.M., Williamson, V. et al. Fully automatic wound segmentation with deep convolutional neural networks. Sci Rep 10, 21897 (2020). https://doi.org/10.1038/s41598-020-78799-w
