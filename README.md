# Hand Gesture Detection
The objective of this project is to identify hand gestures and open specific program assigned to each hand gesture. This project was done using "Deep Learning and Neural Networks". We have used convolutional neural network for this project as they are good for image classification.

We have used Opencv and Keras which is an open source neural network library written in Python. It is capable of running on top of TensorFlow, Microsoft Cognitive Toolkit or Theano.

We have used 3 convolutional layers, 2 dropout layers, 2 maxpooling layer and 1 hidden layer with sigmoid activation funtion and adam as optimiser function. We have also used 'ImageDataGenerator' to increase the number of dataset by manipulating the images.

For this project, we have create our own dataset using a python script and a webcam. Then we created a filter to extract only the hand region from the images, this was done by converting the images form RGB to HSV and then creating a mask for identifying the human skin. Then the images was resized 48x48 and coverted to Grayscale.The dataset can be found at https://drive.google.com/open?id=1oSpLgBYzCJF5gnHgL4nmpOJs0XXESq9W.

We got desired output as notepad was opening when one finger was shown and calculater when 2 finger was shown.

Note: This project is done by team 'Exzodia' of SRM Institute of Science and Technology, Vadaplanai. This team was formed by me
and 2 of my friends Visha Bharthy and Abhishek Nath.

# Original Pic:
![Orginalpic](https://github.com/arupkpatel/HandGestureDetection/blob/master/originalpic.JPG)

# RGB to HSV:
![RGB2HSV](https://github.com/arupkpatel/HandGestureDetection/blob/master/rgb2hsv.JPG)

# Mask:
![mask](https://github.com/arupkpatel/HandGestureDetection/blob/master/mask.JPG)

# Skin_Extraction:
![Skin_Extraction](https://github.com/arupkpatel/HandGestureDetection/blob/master/skinextracted.JPG)

# Results:
![OP1](https://github.com/arupkpatel/HandGestureDetection/blob/master/op1.JPG)
![OP2](https://github.com/arupkpatel/HandGestureDetection/blob/master/op2.JPG)
