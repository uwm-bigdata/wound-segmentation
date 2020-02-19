# 2D Wound Segmentation
This project aims at wound area segmentation from natural images in clinical settings. The architectures tested so far includes: U-Net, MobileNetV2, SegNet, VGG16.
## Install requirements
     pip install -r requirements.txt
     
## Data
The training dataset is built by our lab and collaboration parties. For a sample testing dataset please try the [Medetec Wound Database](http://www.medetec.co.uk/files/medetec-image-databases.html) in the data folder.
    
## Run
    python3 train.py
    python3 predict.py
