import torch
import torchvision.transforms as transform
import torch.nn.functional as F
import pandas as pd
import PIL.Image as Image
import os

''' 
The purpose of this Python script is to run a trained model to identify different vegetable images. To do this, it uses the machine learning library, PyTorch, to run the trained convolutional neural network (CNN).

This script demonstrates how a CNN can be trained to identify qualitative features of images. Given the right training and testing datasets, this tool can be applied to automate qualitative data analysis workflows by identifying image data such as disease states or even trends found in graphical outputs.

To run the model, input unidentified vegetable images from one of the 10 classes into the unidentified_images_directory. Once run, the trained model will output the vegetable classification for each image and display the softmax values (probabilities) for each class to show how certain the model was when classifying a given image.

The images used for this script can be found here: https://www.kaggle.com/datasets/matthewhakar/vegetable-images

* Disclaimer --> Since the best model had an accuracy of 95.89% on the testing dataset after 15 epochs, there is definitely room for improvement. Steps can be taken to further increase the accuracy of the model such as adjusting some of the hyper-parameters for future training sessions. However, the best way to improve the model would be to increase both the number and diversity of training and testing images. A softmax analysis could also be useful for determining where the model is most "confused." For example, the model may have a hard time differentiating between the red pepper and the red tomato images, so this is where most of the focus could then be applied when optimizing the model in the future.
'''

# define the best_model_path (same as the best_model_path from the training script) and the unidentified_images_directory

best_model_path = r'C:\Users\matth\Documents\Matt Hakar\Python\Github Scripts\matthakar_machine_learning\best_model\best_vegetable_id_model.pth'
unidentified_image_directory = r'C:\Users\matth\Documents\Matt Hakar\Python\Github Scripts\matthakar_machine_learning\vegetable dataset\unidentified_images'

# define the classes based on their index position, the index and list position of each class is equal to the alphabetical order of the lists in training and test folders

classes = ['broccoli', 'cabbage', 'carrot', 'cauliflower', 'cucumber', 'pepper', 'potato', 'pumpkin', 'radish', 'tomato']

# define lists for the softmax output dataframe

model = torch.load(best_model_path)

# mean and std from the training dataset

mean = [0.4944, 0.4593, 0.3723]
std = [0.2041, 0.1998, 0.2031]

# transform new image

image_transforms = transform.Compose([transform.Resize((224, 224)), transform.ToTensor(), 
                transform.Normalize(torch.Tensor(mean), torch.Tensor(std))])

# function to classify a single image and show associated softmax values for each class

def classify(model, image_transforms, image_path, classes):
    model = model.eval()
    image = Image.open(image_path).convert('RGB')
    image = image_transforms(image).float()
    image = image.unsqueeze(0)
    output = model(image)
    _, predicted = torch.max(output.data, 1)

    # show softmax (probability values) for each image
    softmax_output = F.softmax(output, dim = 1)
    probabilities = softmax_output.detach().numpy()
    softmax_list = probabilities.tolist()
    softmax_list = softmax_list[0]

    print(' ')
    print(f'image classification for {filename}: {classes[predicted.item()]}')

    # limit softmax outputs to three decimal places
    softmax_list = [f"{num:.3f}" for num in softmax_list]

    # create a dataframe to view the softmax ouputs for each classified image
    classes_df = pd.DataFrame(classes, columns = ['vegetable'])
    softmax_list_df = pd.DataFrame(softmax_list, columns = ['softmax'])
    probability_df = pd.concat([classes_df, softmax_list_df], axis = 1)

    # sort the resulting probability_df by the softmax outputs
    probability_df = probability_df.sort_values(by = 'softmax', ascending = [False])

    print(probability_df.to_string(index=False))

# run the trained model on all images in the unidentified_image_directory

for file in os.listdir(unidentified_image_directory):
    filename = os.fsdecode(file)
    unidentified_image_path = os.path.join(unidentified_image_directory, filename)
    classify(model, image_transforms, unidentified_image_path, classes)
