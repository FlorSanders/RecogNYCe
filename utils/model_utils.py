import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.models import RegNet_Y_1_6GF_Weights, ResNet18_Weights
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def read_csv_data(path):
    # Reading the data and splitting the data
    df = pd.read_csv(path)

    # Discarding irrelevant for classification information (keeping only path to load from and label)
    df = df[['path', 'borough', 'id', "longitude", "latitude"]]
    train_df, test_df = train_test_split(df, random_state = 3, test_size = 0.2)
    val_df, test_df = train_test_split(test_df, random_state = 3, test_size = 0.5)

    return train_df, val_df, test_df

# Defining a custom dataset
class CustomDataset(Dataset):
    def __init__(self, df, test=False):                   
        self.data = []
        for index in range(len(df)):                       
          path = df.iloc[index][0]
          category = df.iloc[index][1]
          self.data.append([path, category])
        self.class_map = {"Bronx" : 0, "Brooklyn": 1,"Manhattan": 2, "Queens": 3, "Staten Island": 4}
        #self.img_dim = (224, 224)
    def __len__(self):                                
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)                    
        #img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])
        return img_tensor, class_id
    
def Training(dataloader, model, transforms, loss_fn, optimizer, dataset = 'imagenet'):
    train_loss = 0                
    num_batches = len(dataloader)
    model.train()                 
    for batch, (X, y) in enumerate(dataloader):   # For each batch
        print(f"{batch} / {num_batches}")
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()                       # Zeroing the gradients
        X = transforms(X)
        if dataset != 'imagenet':
            X = X.to(torch.float32)
        output = model(X)                           # Perform a forward pass
        loss = loss_fn(output, y.squeeze())         # Calculate the loss function

        loss.backward()                             # Perform Backpropagation
        optimizer.step()                            # Update the weights

        train_loss += loss.item()                   # Accumulate the error

    train_loss /= num_batches                     # Average error per epoch
    return train_loss

def Validation(dataloader, model, transforms, loss_fn, dataset = 'imagenet'):

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()                                 
    val_loss, correct = 0, 0

    with torch.no_grad():                         
        for X, y in dataloader:                   
            X = X.to(device)
            y = y.to(device)
            X = transforms(X)
            if dataset != 'imagenet':
                X = X.to(torch.float32)
            pred = model(X)                       
            val_loss += loss_fn(pred, y.squeeze()).item()   
            correct += (pred.argmax(1) == y.squeeze()).type(torch.float).sum().item()

    val_loss /= num_batches         
    correct /= size
    return val_loss, correct


def Train_Validation(model, transforms, train_loader, val_loader, epochs, loss_function, optimizer, dataset = 'imagenet', name = 'resnet18'):

    val_loss_prev = 1000

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = Training(train_loader, model, transforms, loss_function, optimizer, dataset)
        print(f"Training Loss: {train_loss:>8f}")
        val_loss, val_acc = Validation(val_loader, model, transforms, loss_function, dataset)


        if abs(val_loss_prev - val_loss) < 0.005:            # If the change is very small, early stop training
            print("Early Stopping - Epoch: {}".format(t+1))
            break

        else:                                            
            if val_loss < val_loss_prev :
                string = 'weights_{}.pth'.format(name, epochs)
                torch.save(model.state_dict(), string)
                val_loss_prev = val_loss
                print(f"Validation Loss: {val_loss:>8f} \nValidation Accuracy: {(100*val_acc):>0.1f}% (Model Checkpoint)\n")
            else:
                print(f"Validation Loss: {val_loss:>8f} \nValidation Accuracy: {(100*val_acc):>0.1f}%\n")

    print("Training Done!")

def TransferLearning(model, transforms, train_loader, val_loader, lr = 0.001, epochs = 10, replace = False, tune = False, dataset = 'imagenet', name = 'resnet18', opt = 'sgd'):
    if tune == False:
        for param in model.parameters():    # The pretrained parameters are frozen
            param.requires_grad = False
        num_ftrs = model.fc.in_features     # We replace the fc layer
        model.fc = nn.Linear(num_ftrs, 5)

    elif tune == True:                    # Fine-tuning
        if replace == True:
            num_ftrs = model.fc.in_features    
            model.fc = nn.Linear(num_ftrs, 5)
        for param in model.parameters():   
            param.requires_grad = True

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    if opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)

    Train_Validation(model, transforms, train_loader, val_loader, epochs, loss_fn, optimizer, dataset, name)
    #model.load_state_dict(torch.load('model_weights.pth'))  # After training restore the state of the model and evaluate on test data
    #Testing(model, transforms, test_loader)

def Testing(test_loader, trained_model, transforms, display = True, figure_path=None, dataset = 'imagenet'):
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    trained_model.eval()
    correct = 0

    images = []
    true_labels = []
    pred_labels = []
    counter = 0

    all_y = []
    all_y_hat = []

    with torch.no_grad():
        for X, y in test_loader:
            all_y.extend(y)

            if dataset != 'imagenet':
                X = X.to(torch.float32)
            X = X.to(device)
            y = y.to(device)
            X_t = transforms(X)
            pred = trained_model(X_t)
            correct += (pred.argmax(1) == y.squeeze()).type(torch.float).sum().item()
            y_hat = pred.to("cpu")
            all_y_hat.extend(y_hat)


            # Finding falsely classified images to display
            if display == True:
                if counter < 10:
                    wrong_indices = ((pred.argmax(1) == y.squeeze())==False).nonzero()
                    if wrong_indices.squeeze().size() == torch.Size([]):
                        continue
                    elif wrong_indices.squeeze().size()[0] >= 10:
                        for i in range(10):
                            #print(counter, 'all')
                            k = wrong_indices[i].item()
                            images.append(X[k].cpu())
                            if y[k].item() == 0:
                                true_labels.append('Bronx')
                            elif y[k].item() == 1:
                                true_labels.append('Brooklyn')
                            elif y[k].item() == 2:
                                true_labels.append('Manhattan')
                            elif y[k].item() == 3:
                                true_labels.append('Queens')
                            elif y[k].item() == 4:
                                true_labels.append('Staten Island')

                            if pred.argmax(1)[k].item() == 0:
                                pred_labels.append('Bronx')
                            elif pred.argmax(1)[k].item() == 1:
                                pred_labels.append('Brooklyn')
                            elif pred.argmax(1)[k].item() == 2:
                                pred_labels.append('Manhattan')
                            elif pred.argmax(1)[k].item() == 3:
                                pred_labels.append('Queens')
                            elif pred.argmax(1)[k].item() == 4:
                                pred_labels.append('Staten Island')

                            counter += 1
                    else:
                        for i in range(wrong_indices.squeeze().size()[0]):
                            k = wrong_indices[i].item()
                            #print(counter, 'none')
                            images.append(X[k].cpu())
                            if y[k].item() == 0:
                                true_labels.append('Bronx')
                            elif y[k].item() == 1:
                                true_labels.append('Brooklyn')
                            elif y[k].item() == 2:
                                true_labels.append('Manhattan')
                            elif y[k].item() == 3:
                                true_labels.append('Queens')
                            elif y[k].item() == 4:
                                true_labels.append('Staten Island')
                            if pred.argmax(1)[k].item() == 0:
                                pred_labels.append('Bronx')
                            elif pred.argmax(1)[k].item() == 1:
                                pred_labels.append('Brooklyn')
                            elif pred.argmax(1)[k].item() == 2:
                                pred_labels.append('Manhattan')
                            elif pred.argmax(1)[k].item() == 4:
                                pred_labels.append('Queens')
                            elif pred.argmax(1)[k].item() == 5:
                                pred_labels.append('Staten Island')
                            counter += 1

        correct /= size
        print("--------------------")
        print(f"Test Accuracy: {(100*correct):>0.1f}%\n")

        # Plotting 5 falsely classified images
        if display == True:
            fig, axs = plt.subplots(2, 5, figsize=(10,4))
            for i in range(2):
                for j in range(5):
                    if dataset != 'imagenet':
                        axs[i, j].imshow(images[i*5+j].to(torch.int32).permute(1,2,0).numpy()[:,:,::-1])
                    else:
                        axs[i, j].imshow(images[i*5+j].permute(1,2,0).numpy()[:,:,::-1])
                    string = 'True Label = {}\nAssigned Label = {}'.format(true_labels[i*5+j], pred_labels[i*5+j])
                    axs[i, j].set_title(string, fontsize='x-small')
                    axs[i, j].set_xticks([])
                    axs[i, j].set_yticks([])
            if figure_path is not None:
                fig.tight_layout()
                fig.savefig(figure_path, dpi=300, facecolor="white")
            plt.show()
    
    return correct, all_y, all_y_hat

def load_model(model_config, models_dir):
    # Destructure config
    architecture = model_config["architecture"]
    weights_path = os.path.join(models_dir, model_config["weights"])

    # Load models
    if architecture == "benchmark":
        model = create_benchmark_model()
    else:
        model = models.__dict__[architecture]()
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 5)
    
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))    
    if "places" in weights_path or architecture == "benchmark":
        print("Loading Resnet (Places) transforms")
        model_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
    elif "resnet" in weights_path:
        print("Loading Resnet (ImageNet) transforms")
        model_transforms = ResNet18_Weights.IMAGENET1K_V1.transforms(antialias=True)
    elif "regnet" in weights_path:
        print("Loading Regnet (ImageNet) transforms")
        model_transforms = RegNet_Y_1_6GF_Weights.IMAGENET1K_V1.transforms(antialias=True)
    else:
        raise ValueError(f"Transforms for model {weights_path} are unknown")

    return model, model_transforms

def create_benchmark_model():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 6, 5, padding="same"),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(4),
        torch.nn.Conv2d(6, 6, 5, padding="same"),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(4),
        torch.nn.Flatten(),
        torch.nn.Linear(1176, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 5),
    )
    return model
