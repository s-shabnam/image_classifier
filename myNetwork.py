# Imports here
import workspace_utils as wu
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from PIL import Image

image_size = 256
crop_size = 224
normalize_param_mean = [0.485, 0.456, 0.406]
normalize_param_std = [0.229, 0.224, 0.225]
batch_size = 64

input_size = 25088 
#hidden_size = 1024
output_size = 102
dropout_ps = 0.2
epochs_nb = 3
learning_rate = 0.001

checkpoint_name = 'checkpoint'

class Classifier(nn.Module):
    def __init__(self, hidden_units):
        global hidden_size
        hidden_size = hidden_units
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Dropout module with drop probability
        self.dropout = nn.Dropout(p = dropout_ps)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        # Now with dropout
        x = self.dropout(F.relu(self.fc1(x)))      
        # output so no dropout here
        x = F.log_softmax(self.fc2(x), dim=1)
        return x 
    
def set_directories (data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_directories = {'train' : train_dir,
                'valid' : valid_dir,
                'test' : test_dir    
    }
    return data_directories

def transform_data(data_dirs, data_sets):
    data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(crop_size),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(normalize_param_mean, normalize_param_std)
    ]),
    'valid': transforms.Compose([transforms.Resize(image_size),
                                       transforms.CenterCrop(crop_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize(normalize_param_mean, normalize_param_std)
    ]),
    'test': transforms.Compose([transforms.Resize(image_size),
                                       transforms.CenterCrop(crop_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize(normalize_param_mean, normalize_param_std)
        ])
    }

    image_datasets = {data_set : datasets.ImageFolder(data_dirs[data_set], transform = data_transforms[data_set]) for data_set in data_sets } 

    data_loaders = {data_set : torch.utils.data.DataLoader(image_datasets[data_set], batch_size = batch_size, shuffle = True) for data_set in data_sets }

    dataset_sizes = {data_set: len(image_datasets[data_set] )
                   for data_set in data_sets
    }
    return data_loaders, dataset_sizes, image_datasets

def set_architecture(architecture, hidden_units, learning_rate):
    if architecture == 'vgg19':
        model = models.vgg19(pretrained = True)
        global arch
        arch = 'vgg19'
    elif architecture == 'vgg13':
        model = models.vgg13(pretrained = True)  
        arch = 'vgg13'
    elif architecture == 'vgg16':
        model = models.vgg16(pretrained = True)  
        arch = 'vgg16'

    else:
        print('The input models architecture is not recognized')
        
     # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Add rhe new classifier
    model.classifier = Classifier(hidden_units)

    # Prepare training of the classifier layer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    return model, criterion, optimizer

def train_model(data_loaders, model, criterion, optimizer, device, epochs_nb = 10):
    model.to(device);
    with wu.active_session():
        steps = 0
        running_loss = 0
        print_every = 5
        for epoch in range(epochs_nb):
            for inputs, labels in  data_loaders['train']:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in data_loaders['valid']:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)
                            valid_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs_nb}.. "
                          f"Step {steps}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Valid loss: {valid_loss/len(data_loaders['valid']):.3f}.. "
                          f"Valid accuracy: {accuracy/len(data_loaders['valid']):.3f}")
                    running_loss = 0
                    model.train()
    return model

def check_accuracy(model, data_loader, device):  
    model.eval()
    model.to(device)
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model.forward(inputs)
            
            # Calculate accuracy
            top_p, top_class = output.topk(1, dim=1)
            prob = torch.exp(top_p)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
        print(f"Accuracy of the network on the {len(data_loader):.3f}"
          " images: %d %%" % (100 * accuracy/len(data_loader)))
                    
def save_model(image_datasets, model, device, save_directory):
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {'model_arch': arch,
                  'model_device': device,
                  'hidden_units': hidden_size,
                  'class_to_idx' : model.class_to_idx,
                  'state_dict': model.state_dict()}
    save_name = '{}{}_{}.pth'.format(save_directory, checkpoint_name, arch)
    torch.save(checkpoint, save_name)
    print('Model is saved to : ', save_name)
       
def load_model_from_checkpoint(checkpoint_filepath):
    checkpoint = torch.load(checkpoint_filepath)
    if checkpoint['model_arch'] == 'vgg19':
        model = models.vgg19(pretrained = True)
        arch = 'vgg19' 
    elif checkpoint['model_arch'] == 'vgg13':
        model = models.vgg13(pretrained = True)  
        arch = 'vgg13' 
    elif checkpoint['model_arch'] == 'vgg16':
        model = models.vgg16(pretrained = True)  
        arch = 'vgg16'
    else:
        print('The models architecture is not recognized')
    for param in model.parameters():
            param.requires_grad = False   
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = Classifier(checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['state_dict']) 
    #model.to(checkpoint['model_device'])
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''    
    # scale
    imageL = Image.open(image_path)
    image_width = imageL.size[0]
    image_height = imageL.size[1]
    
    if image_width > image_height : # width > height
        imageL.thumbnail((100000, image_size))
    else:
        imageL.thumbnail((image_size, 100000))
    
    # crop
    left_margin = (image_width - crop_size)/2
    bottom_margin = (image_height - crop_size)/2
    right_margin = left_margin + crop_size
    top_margin = bottom_margin + crop_size
    center_box = (left_margin, bottom_margin, right_margin,    
                   top_margin)
    
    imageL.crop(center_box)
    
    # normalize
    np_image = np.array(imageL)/255
    
    mean = np.array(normalize_param_mean)
    std = np.array(normalize_param_std)
    np_image = (np_image - mean)/std
    tronsposed_image = np_image.transpose((2, 0, 1))
    return tronsposed_image

def predict(image_path, model, category_names, device, topk = 5):
    ''' Predict the class (or classes) of an image using a trained deep learning model. '''
    # process image
    model.to(device)
    image = process_image(image_path)
    
    # convert to tensor
    image_tensor = torch.from_numpy(image)# 
    image_tensor = image_tensor.type(torch.FloatTensor)
    image_tensor.resize_([1, 3, crop_size, crop_size])

    #do prediction
    model_output = model(image_tensor.to(device))
    prediction = torch.exp(model_output)
    probabilities, predicted_labels = prediction.topk(topk)
    probabilities, predicted_labels =  probabilities.to(device), predicted_labels.to(device)
    probabilities = probabilities.cpu().detach().numpy().tolist()[0]
    predicted_labels = predicted_labels.cpu().detach().numpy().tolist()[0]

    # get classes
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    idx_to_class = {val: key for key, val in
                                     model.class_to_idx.items()}
    predicted_classes = [idx_to_class[label] for label in predicted_labels]
    predicted_class_labels = [cat_to_name[classes] for classes in predicted_classes]
    return probabilities, predicted_class_labels