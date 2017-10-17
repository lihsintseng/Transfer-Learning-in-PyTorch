### Transfer Learning in PyTorch
##### Author: Li-Hsin Tseng
Source code: train.py, test.py

###### Part A: train.py

This part I tried to use a pretrained [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) model and trained the last layer(fully connected layer) to fit on to the [Tiny ImageNet](https://tiny-imagenet.herokuapp.com) dataset, which contains 200 categories.

In this file are two classes, AlexNet() and LastLayer(). 
AlexNet() defines the first part of the new network which I removed the last fully connected layer so that I can then use its output as input of the LastLayer(). These two classes together forms a whole network.

```python
python3 train.py --data data/tiny-imagenet-200/ --save model/my_model.pt
```

Using the above comment can let you use your own dataset and you can save the last layer model for later usage.

I trained the last layer for over 200 times and the training loss reduced from over 2 to 0.4015. However, for testing accuracy, it is always around 0.0039 and never goes up.

###### Part B: test.py

```python
python3 test.py --model model/my_model.pt
```
By using the above comment, which the directory stores the last layer model. 

By running the script, it would take frames from the camera and use the Alexnet and the last layer model to predict what the frame may contains.

The classes' names for the Tiny imageNet is quite odd and one can find the corresponding category in the words.txt file.

