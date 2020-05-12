# Age Estimation
This is a PyTorch implementation of mean-variance loss [1] and softmax loss embedded into ResNet18 for age estimation.

## Dependencies
- Python 3.7+
- PyTorch 1.0+
- Pillow 5.4+
- NumPy 1.16+
- TorchVision 0.2+
- OpenCV 4.1.1+

Tested on:
- Ubuntu 18.04, CUDA 10.1
- CPU: Intel Broadwell, GPU: NVIDIA Tesla K80

## Usage
### Setting
- I follow a widely used leave-one-person-out (LOPO) protocol in my experiments. Images of a person is used as test data, images of the others is used as training and validation data. Ratio of validation data to training and validation data is set as "VALIDATION_RATE= 0.1" in "main.py".

### Training
1. Prepare a FG-NET dataset.
DataLoaders of this program load images from a FG-NET dataset. A image filename of the FG-NET is prefixed by a subject index and contains a age label. Thus subject indices and labels are extracted from filenames.

URL: http://yanweifu.github.io/FG_NET_data/FGNET.zip

2. Start training

Mean-Variance Loss
```
python main.py --batch_size 64 --image_directory FGNET/images --leave_subject 1 --learning_rate 0.001 --epoch 100 --result_directory result/mean_variance -loss

--leave_subject: a subject index for test data (1~82). (integer)
--result_directory: a directory where the model will be saved.
-loss: mean-variance loss
```

Only Softmax Loss
```
python main.py --batch_size 64 --image_directory FGNET/images --leave_subject 1 --learning_rate 0.001 --epoch 100 --result_directory result/softmax

--leave_subject: a subject index for test data (1~82). (integer)
--result_directory: a directory where the model will be saved.
```

### Comparison of mean-variance loss with softmax cross entropy
The following figure shows Mean Absolute Error (MAE) when the subject #1 is chosen as test data. We see that MAE on validation and test data of mean-variance loss and softmax cross entropy loss is better than that of softmax cross entropy loss only.
![Figure_1](https://github.com/Herosan163/AgeEstimation/blob/images/mae.png)

### Inference
```
python main.py -pi FGNET/images/001A43a.JPG -pm result/model_best_loss
```
The following figures show the results of the subject #1 who I choosed as test data.
The left image is the picture when the subject was 2 years old.
The right image is the picture when the subject was 43 years old.

![001A02_result](https://github.com/Herosan163/AgeEstimation/blob/images/1.png)
![001A43a_result](https://github.com/Herosan163/AgeEstimation/blob/images/2.png)

### Differences between this implementation and original a paper.
- In the original paper, AlexNet and VGG-16 are used as feature extractors. On the other hand, ResNet18 is used in this program.
- In the original paper, SGD optimizer is used. On the other hand, Adam optimizer is used in this program.
- In the original paper, all the face images are aligned based on five facial landmarks detected using an open-source SeetaFaceEngine [2]. In this program, no alignment is done.
- In this program, three augmentations are adopted to images randomly, i.e., horizontal flip, rotate and shear.

## References
[1] H. Pan, et al. "Mean-Variance Loss for Deep Age Estimation from a Face." Proceedings of CVPR, 2018.

[2] https://github.com/seetaface/SeetaFaceEngine
