import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transform
import torchvision.models as models
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

''' 
The purpose of this Python script is to train a model to identify different vegetable images. To do this, it uses the machine learning library, PyTorch, to train and test a convolutional neural network (CNN) on over 12,000 images.

This script demonstates how a CNN can be trained to identify qualitative features of images. Given the right training and testing datasets, this tool can be applied to automate qualitative data analysis workflows by identifying image data such as disease states or even trends found in graphical outputs.

The images used for this script can be found here: https://www.kaggle.com/datasets/matthewhakar/vegetable-images

This script is split into 4 main parts after the data and model paths are defined:

Part 1. Calculate the mean and standard deviation (std) of the training images for normalization transforms
Part 2. Apply transforms to all training and testing images. Show a sample of these images
Part 3. Load data and run the ResNet18 convolutional neural network (CNN). Train the model using the training dataset, and display the accuracy on both the training and testing datasets after each epoch
Part 4. Save the machine learning model that showed the best accuracy on the testing dataset. This is the final model for a given training session
'''

# define training and testing dataset paths

training_dataset_path = r'C:\Users\matth\Documents\Matt Hakar\Python\Github Scripts\matthakar_machine_learning\vegetable dataset\training'
testing_dataset_path  = r'C:\Users\matth\Documents\Matt Hakar\Python\Github Scripts\matthakar_machine_learning\vegetable dataset\testing'

# define the checkpoint and best model paths

checkpoint_path = r'C:\Users\matth\Documents\Matt Hakar\Python\Github Scripts\matthakar_machine_learning\best_model\best_vegetable_id_checkpoint.pth.tar'
best_model_path = r'C:\Users\matth\Documents\Matt Hakar\Python\Github Scripts\matthakar_machine_learning\best_model\best_vegetable_id_model.pth'

'''
Part 1. Calculate the mean and std of the training images for normalization transforms
'''

# apply transforms to the training images (resizes and converts each image to a tensor before calculations are run)

training_transforms = transform.Compose([transform.Resize((224, 224)), transform.ToTensor()])
train_dataset = torchvision.datasets.ImageFolder(root = training_dataset_path, transform = training_transforms)

# load data and define batch_size

train_loader = torch.utils.data.DataLoader (dataset = train_dataset, batch_size = 64, shuffle = False)

# calculate the mean and std of training image pixels for normalization

def calculate_mean_and_std(loader):
    # define the mean and std as floats
    mean = 0.
    std = 0.
    # define total_images_count as an integer
    total_images_count = 0
    for images, _ in loader:
        image_count_in_a_batch = images.size(0)
        images = images.view(image_count_in_a_batch, images.size(1),-1)
        # calculate mean and std of each image batch
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += image_count_in_a_batch
    # update the mean and std by dividing the sum of the mean and std for each image batch by the total image count
    mean /= total_images_count 
    std /= total_images_count
    return mean, std

mean, std = calculate_mean_and_std(train_loader)

# print the mean and std results for normalization

print(' ')
print('mean and std for normalization:')
print(' ')
print(f'mean = [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]')
print(f'std = [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]')
print(' ')

'''
Part 2. Apply transforms to all training and testing images. Show a sample of these images
'''
# add mean and std output from Part 1 to lists for normalization:

mean = [mean[0], mean[1], mean[2]]
std = [std[0], std[1], std[2]]

# apply transforms and normalize data for both the training and testing datasets, transforms such as rotation and flip increase data diversity while others like resize and normalize prevent data leakage

training_and_testing_transforms = transform.Compose([transform.Resize((224, 224)), transform.RandomRotation(degrees = 45), 
transform.RandomHorizontalFlip(), transform.RandomVerticalFlip(), transform.ToTensor(), transform.Normalize(torch.Tensor(mean), torch.Tensor(std))])
train_dataset = torchvision.datasets.ImageFolder(root = training_dataset_path, transform = training_and_testing_transforms)
test_dataset = torchvision.datasets.ImageFolder(root = testing_dataset_path, transform = training_and_testing_transforms)

# show an example of the images after the transforms are applied

def show_sample_of_transformed_images(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size = 6, shuffle = True)
    batch = next(iter(loader))
    images, labels = batch
    grid = torchvision.utils.make_grid(images, nrow = 3)
    plt.figure(figsize=(11, 11))
    plt.imshow(np.transpose(grid, (1,2,0)))
    plt.show()
    # show what class the sample images belong to (each class is assigned an index number) 
    print('class index numbers:', labels) 

show_sample_of_transformed_images(train_dataset) 

'''
Part 3. Load data and run the ResNet18 convolutional neural network (CNN). Train the model using the training dataset, and display the accuracy on both the training and testing datasets 
'''

# load the train_dataset and the test_dataset, only the train_dataset needs to be shuffled

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle = False)

# run the CNN with the GPU if available

def set_device():
    if torch.cuda.is_available():
        dev = 'cuda:0'
    else:
        dev = 'cpu'
    return torch.device(dev)

# define function to save the best checkpoint

def save_checkpoint(model, epoch, optimizer, best_accuracy):
    state = {'epoch': epoch + 1, 'model': model.state_dict(), 'best_accuracy' : best_accuracy, 'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)

# define training loop function

def train_the_model(model, train_loader, test_loader, criterion, optimizer, number_of_epochs):
    device = set_device()
    best_accuracy = 0
    # run every epoch in the number_of_epochs hyper-parameter
    for epoch in range (number_of_epochs):
        print('epoch number %d'% (epoch + 1))
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0
        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            # reset gradient back to zero with .zero_grad()
            optimizer.zero_grad()
            outputs = model(images)
            _, predicted = torch.max(outputs.data,1)
            loss = criterion(outputs, labels)
            # backpropagate to calculate the weight gradient
            loss.backward() 
            # update weights
            optimizer.step() 
            running_loss += loss.item()
            running_correct += (labels == predicted).sum().item()
        # calculate epoch_loss and epoch_accuracy for each epoch
        epoch_loss = running_loss/len(train_loader)
        epoch_accuracy = 100.00 * (running_correct / total)
        # print the number of training images correctly classified in each epoch and the associated loss
        print(' - training dataset- got %d out of %d images correct (%.3f%%) epoch loss: %.3f'% (running_correct, total, epoch_accuracy, epoch_loss))
        test_dataset_accuracy = evaluate_model_on_test_set(model, test_loader)
        # continuously define the best model as the one with the best accuracy on the test dataset 
        if (test_dataset_accuracy>best_accuracy):
            best_accuracy=test_dataset_accuracy
            save_checkpoint(model, epoch, optimizer, best_accuracy)
    print('finished')
    return model

# define testing loop function

def evaluate_model_on_test_set(model, test_loader):
    model.eval()
    predicted_correctly_on_epoch = 0
    total = 0
    device = set_device()
    # prohibits back propagation (don't want to train on test dataset) and speeds it up
    with torch.no_grad(): 
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            outputs = model(images)
            _, predicted = torch.max(outputs.data,1)
            predicted_correctly_on_epoch += (predicted == labels).sum().item()
    # calculate epoch_accuracy for each epoch
    epoch_accuracy = 100.00 * (predicted_correctly_on_epoch / total)
    # print the number of testing images correctly classified in each epoch
    print(' - testing dataset- Got %d out of %d images correct (%.3f%%)'% (predicted_correctly_on_epoch, total, epoch_accuracy))
    return epoch_accuracy

# load the resnet18 model for training

resnet18_model = models.resnet18(weights=None)
number_of_features = resnet18_model.fc.in_features
number_of_classes = 10
resnet18_model.fc = nn.Linear(number_of_features, number_of_classes)
device = set_device()
resnet18_model = resnet18_model.to(device) 

# define the loss_function

loss_function = nn.CrossEntropyLoss()

# define learning_rate and number_of_epochs hyper-parameters (can change based on needs)

learning_rate = 0.01
number_of_epochs = 15

# define the optimizer, weight decay tries to prevent overfitting

optimizer = optim.SGD(resnet18_model.parameters(), lr = learning_rate, momentum = 0.9, weight_decay = 0.003) 

# run the convolutional neural network on both the training and the test datasets

train_the_model(resnet18_model, train_loader, test_loader, loss_function, optimizer, number_of_epochs)

'''
Part 4. Save the machine learning model that showed the best accuracy on the testing dataset. This is the final model for a given training session
'''
# load the checkpoint that you saved to see details about it

checkpoint = torch.load(checkpoint_path)

print(checkpoint['epoch'])
print(checkpoint['best_accuracy'])

# load the best model

resnet18_model = models.resnet18()
number_of_features = resnet18_model.fc.in_features
number_of_classes = 10
resnet18_model.fc = nn.Linear(number_of_features, number_of_classes)
resnet18_model.load_state_dict(checkpoint['model'])

# save best model

torch.save(resnet18_model, best_model_path)












































































































