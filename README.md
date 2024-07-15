# matthakar_machine_learning

These Python scripts use the PyTorch library to show how a machine learning model can be trained and implemented to identify qualitative features of images. Given the right training and testing datasets, this tool can be applied to automate qualitative data analysis workflows by identifying image data such as disease states or even trends found in graphical outputs.

__________________________________________________________________________________________________________________________________________________________

Script 1: matthakar_machine_learning_training.py

The purpose of this Python script is to train a model to identify different vegetable images. To do this, it uses the machine learning library, PyTorch, to train and test a convolutional neural network (CNN) on over 12,000 images.

The images used for this script can be found here: https://www.kaggle.com/datasets/matthewhakar/vegetable-images

This script is split into 4 main parts after the data and model paths are defined:

Part 1. Calculate the mean and standard deviation (std) of the training images for normalization transforms

Part 2. Apply transforms to all training and testing images. Show a sample of these images

Part 3. Load data and run the ResNet18 convolutional neural network (CNN). Train the model using the training dataset, and display the accuracy on both the training and testing datasets after each epoch

Part 4. Save the machine learning model that showed the best accuracy on the testing dataset. This is the final model for a given training session

__________________________________________________________________________________________________________________________________________________________

Script 2: matthakar_machine_learning_use_case.py

The purpose of this Python script is to run a trained model to identify different vegetable images. To do this, it uses the machine learning library, PyTorch, to run the trained convolutional neural network (CNN).

To run the model, input unidentified vegetable images from one of the 10 classes into the unidentified_images_directory. Once run, the trained model will output the vegetable classification for each image and display the softmax values (probabilities) for each class to show how certain the model was when classifying a given image.

The images used for this script can be found here: https://www.kaggle.com/datasets/matthewhakar/vegetable-images

* Disclaimer --> Since the best model had an accuracy of 95.89% on the testing dataset after 15 epochs, there is definitely room for improvement. Steps can be taken to further increase the accuracy of the model such as adjusting some of the hyper-parameters for future training sessions. However, the best way to improve the model would be to increase both the number and diversity of training and testing images. A softmax analysis could also be useful for determining where the model is most "confused." For example, the model may have a hard time differentiating between the red pepper and the red tomato images, so this is where most of the focus could then be applied when optimizing the model in the future.
