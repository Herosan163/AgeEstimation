# Age Estimation
This is a PyTorch implementation of mean-variance loss [1] and softmax loss embedded into ResNet34 for age estimation.

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
- To converge less epoch, in the first 10 epochs, parameters of only full connected layers is updated. 

### Training
1. Prepare a FG-NET dataset.
DataLoaders of this program load images from a FG-NET dataset. A image filename of the FG-NET is prefixed by a subject index and contains a age label. Thus subject indices and labels are extracted from filenames.

URL: http://yanweifu.github.io/FG_NET_data/FGNET.zip

2. Start training
```
python main.py --batch_size 64 --image_directory FGNET/images --leave_subject 1 --learning_rate 0.002 --epoch 100 --result_directory result

--leave_subject: a subject index for test data (1~82). (integer)
--result_directory: a directory where the model will be saved.
```

### Comparison of mean-variance loss with softmax cross entropy
The following figure shows Mean Absolute Error (MAE) when the subject #1 is chosen as test data. We see that MAE on validation and test data of mean-variance loss and softmax cross entropy loss is better than that of softmax cross entropy loss only.
![Figure_1](https://user-images.githubusercontent.com/53385884/62007786-ae573780-b18c-11e9-9b87-b56bc4b6a014.png)

### Inference
```
python main.py -pi FGNET/images/001A43a.JPG -pm result/model_best_loss
```
The following figures show the results of the subject #1, who I choosed as test data.
![001A02_result](https://user-images.githubusercontent.com/53385884/63864383-b522d580-c9ea-11e9-88bb-096a77793d38.jpg)
![001A43a_result](https://user-images.githubusercontent.com/53385884/63864419-c5d34b80-c9ea-11e9-91c4-d0474d6a7e11.jpg)

### Differences between this implementation and original a paper.
- In the original paper, AlexNet and VGG-16 are used as feature extractors. On the other hand, ResNet34 is used in this program.
- In the original paper, all the face images are aligned based on five facial landmarks detected using an open-source SeetaFaceEngine [2]. In this program, no alignment is done.
- In this program, three augmentations are adopted to images randomly, i.e., horizontal flip, rotate and shear.

## References
[1] H. Pan, et al. "Mean-Variance Loss for Deep Age Estimation from a Face." Proceedings of CVPR, 2018.

[2] https://github.com/seetaface/SeetaFaceEngine
