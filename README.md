# Generate Synthetic Car Images using PGGAN Model

Guide to training a Progressively Growing GAN model on the Car Connection Picture Dataset from Kaggle.  

This repo provides the Python code used for preparing the data to be trained by the official PGGAN implementation. 

This implementation can be used to generate car images at 256p resolution due to time, compute, and data quality constraints. 

## Built with
- [Python](https://www.python.org/) v3.6
- [ImageAI](https://github.com/OlafenwaMoses/ImageAI) v2.1.6
- [NumPy](https://numpy.org/) v1.13.3
- [PIL](http://www.pilofficial.com/) v7
- [cv2](https://pypi.org/project/opencv-python/) v4.5.1.48
- [Tensorflow](https://www.tensorflow.org/install/pip) v1.15.2
- Google Colab Pro leveraging an NVIDIA v100 GPU

## Getting started

### Data Preparation
1. Download dataset from Kaggle [here].(https://www.kaggle.com/prondeau/the-car-connection-picture-dataset)
2. Upload the compressed dataset to Google Drive.
3. Create new Google Colab notebook.
4. Mount Google Drive using the following Python expression.
```
from google.colab import drive
drive.mount('/content/drive')
```
5. Set working directory using bash commands. 

6. Clone this repo using the following expression. 
```
git clone https://github.com/a42labs/Car-GAN-Demo.git
```
7. Use the package manager [pip](https://pip.pypa.io/en/stable/) to install ImageAI Library.
```
pip install imageai
```
8. Choose desired object detection model and upload to working directory. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ImageAI supports 3 popular models. Look at [ImageAI detection class](https://imageai.readthedocs.io/en/latest/detection/) for more documentation.   

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Download RetinaNet Model (high accuracy, low speed)](https://github.com/OlafenwaMoses/ImageAI/releases/download/essentials-v5/resnet50_coco_best_v2.1.0.h5/)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Download YOLOv3 Model (medium accuracy, medium speed)](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5/)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Download TinyYOLOv3 Model (low accuracy, high speed)](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo-tiny.h5/)

9. Locate and unzip the car dataset. 
```
!unzip path_from_data.zip -d path_to_unzip
```
10. Change file dependencies for input, output and pre-trained object detection model.
```
# directory that contains all images
inputdir = '/Users/PGGAN/thecarconnectionpicturedataset/'
# output directory
outputdir = '/Users/PGGAN/processeddataset/'
```
```
detector.setModelPath('/Users/PGGAN/resnet50_coco_best_v2.1.0.h5')
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Desired image resolution can also be changed and the images will be processed accordingly. 


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;To run the file simply call the `load_cars` function, passing in `inputdir` and `outputdir`.

### Model training
1. We will now clone the official progressive growing of GANs repo. 
```
git clone https://github.com/tkarras/progressive_growing_of_gans.git 
```
2. The dataset_tool.py from the PGGAN repo provides a CLI to generate the tfrecords used for training. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Run this tool in a Colab cell, specifying input image directory and output tfrecord directory. 

``` 
!python dataset_tool.py create_from_images /Users/PGGAN/outputdir /Users/PGGAN/inputdir
```
3. To configure the network, specify paths at the top of the `config.py` file. 
```
data_dir = 'datasets'
result_dir = 'results'
```
4. One last work around before training... In the config.py file, you must choose uncomment a dataset from the given list. The `-syn1024rgb` provided for custom datasets does not seem to work. I simply chose the `-lsun-car` dataset option instead, assuming both would have similar hyper parameters. 

5. Run `train.py` to begin training. 

## Usage


