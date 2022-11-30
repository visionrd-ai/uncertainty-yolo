






# Uncertainty Estimation for Object Detectors

## Introduction
Object detectors usually just give out confidence scores with the detections and they usually are not the measurement of how sure a model is in classifying or localizing that object. In this approach we are trying to gain a measurement of how sure a model is in estimating the probability that an object belongs to a certain class of interest (i.e. semantic uncertainity).

##  Epistemic Uncertainty Estimation
For each element of the anticipated anchors $\mu$, and the standard deviation $\sigma$, comprise the Gaussian Probability Distribution function. When testing, we can calculate the Negative log-likelihood that an object is detected and its probability distribution vector $I$ belongs to the class $\hat l$ by:

$$ p = -log( \sum_{j=1}^{M} N(I;\mu_{i,j},\sigma_{i,j}) $$

We compute the log-likelihood of the data $I$ for each known class model to obtain an epistemic uncertainty metric for each known class $\hat l$.

$$ P = (-log(p(I:\hat l_{1})),....,(-log(p(I:\hat l_{N}))  $$

A low negative log-likelihood represents a low uncertainty the detected object belongs to the respective known class.

## Installation
1. Clone repo and install [`requirements.txt`](https://github.com/visionrd-ai/uncertainty-yolo/blob/main/requirements.txt) in a [Python>=3.7.0](https://www.python.org/) environment, including [PyTorch>=1.7](https://pytorch.org/get-started/locally/).
	
	```bash

	git clone https://github.com/visionrd-ai/uncertainty-yolo # clone

	cd uncertainty-yolo

	pip install -r requirements.txt # install

	```
	
2. For this repo, we have used [YOLOv5](https://github.com/ultralytics/yolov5) by [Ultralytics](https://ultralytics.com/). You can head over to their repo if you face any issues with the installation.

## Inference
[`detect.py`](https://github.com/visionrd-ai/uncertainty-yolo/blob/main/detect.py) runs inference on a variety of sources, downloading pre-trained models automatically from the latest [YOLOv5](https://github.com/ultralytics/yolov5) release and saving results to `runs/detect`. 

```bash
python detect.py --source 0  # webcam
                          img.jpg  # image
                          vid.mp4  # video
                          screen  # screenshot
                          path/  # directory
                          'path/*.jpg'  # glob
                          'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                          'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

## Replicating the approach for other models
We have cloned the code from [YOLOv5](https://github.com/ultralytics/yolov5) and added our approach into it. We only modified [`detect.py`](https://github.com/visionrd-ai/uncertainty-yolo/blob/main/detect.py) and [`utils/general.py`](https://github.com/visionrd-ai/uncertainty-yolo/blob/main/utils/general.py). We modified the [`non_max_suppression`](https://github.com/visionrd-ai/uncertainty-yolo/blob/main/utils/general.py#L884) function in order to return the probabilities of all classes with every bounding box. We then added a custom [`calculate_uncertainty`](https://github.com/visionrd-ai/uncertainty-yolo/blob/main/utils/general.py#L1008) function to calculate uncertainties and then plot them with confidences and bounding boxes.
## Future Work
### Prior Probabilities
In order to improve this work, prior probabilities of the same object can be used to calculate uncertainties.
### Normalization of Uncertainties
We are working to normalize the measure of uncertainties so that they can be compared object-wise as well as class-wise during inference.

## Use cases
### Scenario Extraction - Coming Soon on our GitHub!:rocket:
This approach can also be used for scenarios extraction i.e to track the instances/objects where the model will fail. Some examples of advanced use cases of the above mentioned approach are given below:
>An unusual pickup truck is being detected by the model but it is uncertain on it.
![s2_i](https://user-images.githubusercontent.com/110380622/204512223-7ecf8f91-9676-4cab-9c31-440aff8dddd4.gif)
![s2_o](https://user-images.githubusercontent.com/110380622/204512297-79ca8286-fbb0-40de-a30c-4615d7280328.gif)

>An unusual bike carrying sacks is being detected by the model but it is uncertain on it.
![s5_i](https://user-images.githubusercontent.com/110380622/204512483-297eecef-4c55-43f6-9a29-61bd3ad05bbb.gif)![s5_o](https://user-images.githubusercontent.com/110380622/204512540-0b5f5500-d271-47a2-b0fb-7e3be0a30597.gif)

### Class-wise Instances Uncertainty Analysis
![vehicle](https://user-images.githubusercontent.com/110380622/204519138-b66a7081-7ab0-4841-be51-a116dde90362.png)
![person](https://user-images.githubusercontent.com/110380622/204519112-e2e6c48e-5af3-4a31-bb6f-6d5f0195ffbf.png)

It is indispensable for companies working in autonomous driving to capture uncertainties to make their autonomous vehicles safe for public use. Consider an autonomous vehicle operating in an Asian environment where it encounters a rickshaw or a tricycle etc. which are quite comman in Asian environments but are very rare in western environments where most of the models are trained. The models will become uncertain and there will be a huge risk of failure. This failure is very dangerous for companies in autonomous sector therefore we see companies like [Bosch](https://www.bosch.com/),  [Audi](https://www.audi.com/en.html) etc. investing heavily in gathering data from Asian and African environments.
## Acknowledgements

 - We thank [Ultralytics](https://ultralytics.com/) for their awesome work on [YOLOv5](https://github.com/ultralytics/yolov5).
 - *[Uncertainty for Identifying Open-Set Errors in Visual Object Detection](https://arxiv.org/pdf/2104.01328.pdf)*

------------
[<img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/96cafbd73ec4339b8c73c36673ce1518db82cc5c/svgs/brands/linkedin.svg" width="30" height="30">](https://www.linkedin.com/company/visionrd-ai/) [<img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/96cafbd73ec4339b8c73c36673ce1518db82cc5c/svgs/brands/github.svg" width="30" height="30">](https://github.com/visionrd-ai)

Please follow and visit us [VisionRD](http://visionrdai.com/)!
