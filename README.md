# 2D Wound Segmentation
This project aims at wound area segmentation from natural images in clinical settings. The architectures tested so far includes: U-Net, MobileNetV2, Mask-RCNN, SegNet, VGG16.
![Intro_Image](https://raw.githubusercontent.com/Pele324/ChronicWoundSeg/master/figures/Intro.png)
![Dataset_Image](https://raw.githubusercontent.com/Pele324/ChronicWoundSeg/master/figures/Dataset.png)

## Publication
C. Wang, D.M. Anisuzzaman, V. Williamson, M.K. Dhar, B. Rostami, J. Niezgoda, S. Gopalakrishnan, and Z. Yu, “Fully Automatic Wound Segmentation with Deep Convolutional Neural Networks”, Scientific Reports, 10:21897, 2020. https://doi.org/10.1038/s41598-020-78799-w

## Data
The training dataset is built by our lab and collaboration clinic, Advancing the Zenith of Healthcare (AZH) Wound and Vascular Center. With their permission, we are sharing this dataset (./data/wound_dataset/) publicly. This dataset was fully annotated by wound professionals and preprocessed with cropping and zero-padding.  
  
## Updates 
3/12/2021:  
The dataset is now available as a MICCAI online challenge. The training and validation dataset are published [here](https://github.com/uwm-bigdata/wound-segmentation/tree/master/data/Foot%20Ulcer%20Segmentation%20Challenge) and we will start evaluating on the testing dataset in August 2021. Please find more details about the challenge [here](http://www.miccai.org/events/challenges/) and [here](https://fusc.grand-challenge.org/).  
  
12/13/2021:  
The code is updated to be compatible with tf 2.x

## Requirements
tensorflow==2.6.0
    
## Run
    python3 train.py
    python3 predict.py
  
## Credits
Thanks for the [AZH Wound and Vascular Center](https://azhcenters.com/) for providing the data and great help with the annotations. MobileNetV2 is forked from [here](https://github.com/bonlime/keras-deeplab-v3-plus). Some utility code are forked from [here](https://github.com/Yt-trium/DeepSeg3D/tree/master/utils).
