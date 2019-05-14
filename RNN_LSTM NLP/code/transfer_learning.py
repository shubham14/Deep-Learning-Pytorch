import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

def train_model(device, dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
   
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # If there is no training happening
    if num_epochs == 0:
        model.eval()
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # statistics
            running_corrects += torch.sum(preds == labels.data)

        best_acc = running_corrects.double() / dataset_sizes['val']

    # Training for num_epochs steps
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                ####################################################################################
                #                             START OF YOUR CODE                                   #
                ####################################################################################
                # Perform feedforward operation using model, get the labels using torch.max, and   #
                # compule loss using the criterion function                                        #
                ####################################################################################
                    out = model(inputs)
                    _, preds = torch.max(out, 1)
                    loss = criterion(out, labels)
    
                # backward + optimize only if in training phase
                if phase == 'train':
                	loss.backward()
                	optimizer.step()

                ####################################################################################
                #                             END OF YOUR CODE                                     #
                ####################################################################################
                    
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    if num_epochs > 0:
        model.load_state_dict(best_model_wts)
    return model


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.savefig('tmp.png')
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(device, dataloaders, model, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            ####################################################################################
            #                             START OF YOUR CODE                                   #
            ####################################################################################
            # Perform feedforward operation using model and get the labels using torch.max     #
            ####################################################################################
            out = model(inputs)
            _, preds = torch.max(out, 1)

            ####################################################################################
            #                             END OF YOUR CODE                                     #
            ####################################################################################
            
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def finetune(device, dataloaders, dataset_sizes, class_names):
    model_ft = models.resnet18(pretrained=True).to(device)
    

    ####################################################################################
    #                             START OF YOUR CODE                                   #
    ####################################################################################
    # Replace last layer in with a 2-label linear layer                                #
    ####################################################################################
    num_filters = model_ft.fc.in_features
    out_features = model_ft.fc.out_features
    model_ft.fc = nn.Linear(num_filters, out_features)
    
    model_ft = model_ft.to(device)

    ####################################################################################
    #                             END OF YOUR CODE                                     #
    ####################################################################################


    ####################################################################################
    #                             START OF YOUR CODE                                   #
    ####################################################################################
    # Set the criterion function for multi-class classification, and set the optimizer #
    ####################################################################################
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9) 

    ####################################################################################
    #                             END OF YOUR CODE                                     #
    ####################################################################################

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Show the performance with the pretrained-model, not finetuned yet
    print('Performance of pre-trained model without finetuning')
    _ = train_model(device, dataloaders, dataset_sizes, model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=0)

    # Finetune the model for 25 epoches
    print('Finetune the model')
    model_ft = train_model(device, dataloaders, dataset_sizes, model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

    visualize_model(device, dataloaders, model_ft, class_names)

def freeze(device, dataloaders, dataset_sizes, class_names):
    model_conv = torchvision.models.resnet18(pretrained=True).to(device)

    ####################################################################################
    #                             START OF YOUR CODE                                   #
    ####################################################################################
    # Freeze all parameterws in the pre-trained network.                               #
    # Hint: go over all parameters and set requires_grad to False                      #
    ####################################################################################
    for param in model_conv.parameters():
    	param.requires_grad = False

    ####################################################################################
    #                             END OF YOUR CODE                                     #
    ####################################################################################


    ####################################################################################
    #                             START OF YOUR CODE                                   #
    ####################################################################################
    # Replace last layer in with a 2-label linear layer                                #
    ####################################################################################
    # Parameters of newly constructed modules have requires_grad=True by default
    num_filters = model_conv.fc.in_features
    out_features = model_conv.fc.out_features
    model_conv.fc = nn.Linear(num_filters, out_features)
    						
    ####################################################################################
    #                             END OF YOUR CODE                                     #
    ####################################################################################


    ####################################################################################
    #                             START OF YOUR CODE                                   #
    ####################################################################################
    # Set the criterion function for multi-class classification, and set the optimizer.#
    # Note: Make sure that the optimizer only updates the parameters of the last layer #
    ####################################################################################
    criterion = nn.CrossEntropyLoss()
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
    
    ####################################################################################
    #                             END OF YOUR CODE                                     #
    ####################################################################################


    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    print('Performance of pre-trained model without finetuning')
    _ = train_model(device, dataloaders, dataset_sizes, model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=0)

    print('Finetune the model')
    model_conv = train_model(device, dataloaders, dataset_sizes, model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)

    visualize_model(device, dataloaders, model_conv, class_names)

def main():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                   shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Finetune the pre-trained model")
    finetune(device, dataloaders, dataset_sizes, class_names)
    print("Freeze the parameters in pre-trained model and train the final fc layer")
    freeze(device, dataloaders, dataset_sizes, class_names)

if __name__== "__main__":
    main()
