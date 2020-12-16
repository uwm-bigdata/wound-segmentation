# 2D Wound Segmentation
This project aims at wound area segmentation from natural images in clinical settings. The architectures tested so far includes: U-Net, MobileNetV2, Mask-RCNN, SegNet, VGG16.
![Intro_Image](https://raw.githubusercontent.com/Pele324/ChronicWoundSeg/master/figures/Intro.png)
![Dataset_Image](https://raw.githubusercontent.com/Pele324/ChronicWoundSeg/master/figures/Dataset.png)

## Publication
Wang, C., Anisuzzaman, D.M., Williamson, V. et al. Fully automatic wound segmentation with deep convolutional neural networks. Sci Rep 10, 21897 (2020). https://doi.org/10.1038/s41598-020-78799-w

## Data
The training dataset is built by our lab and collaboration clinic, Advancing the Zenith of Healthcare (AZH) Wound and Vascular Center. With their permission, we are sharing this dataset (./data/wound_dataset/) publicly. This dataset was fully annotated by wound professionals and preprocessed with cropping and zero-padding. We plan to publish the raw images and annotations as a segmentation challenge of MICCAI 2021.
    
## Run
    python3 train.py
    python3 predict.py
