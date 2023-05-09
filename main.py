import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.modules.loss import BCEWithLogitsLoss
from torchvision import models, datasets

from torchvision.models import ResNet50_Weights

from model_data.resnet50 import ResNet50

from dataset import get_dataset
from transform_data import transform_train, transform_val, transform_test
from help_functions import iterate_dataloader, imshow, get_best_loss
from training import train_model

import lightning.pytorch as pl
import os
#from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import ModelCheckpoint

if __name__ == '__main__':
    # used device for computing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device, 'will be used.')

    # get the dataset
    train_dataset, val_dataset, test_dataset = get_dataset()

    # apply transforms and preprocessing to dataset
    train_dataset.set_transform(transform_train)
    val_dataset.set_transform(transform_val)
    test_dataset.set_transform(transform_test)

    # load data
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # showcase some training images
    images, labels = iterate_dataloader(train_loader, device)
    imshow(images, labels)

    # define model
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # freeze all parameters
    for params in model.parameters(): params.requires_grad_ = False

    # add new final layer to customize model to become binary classifier
    nr_filters = model.fc.in_features  # number of input features of last layer
    model.fc = nn.Linear(nr_filters, 1)

    # load trained model if there is one
    try:
        model = torch.jit.load('./model_data/resnet50_ds1.pth')
        with torch.no_grad():
            print(model)
    except ValueError:
        print("There was no model to load.")

    model = model.to(device)

    # loss; binary cross entropy with sigmoid, i.e. no need to use sigmoid in model
    loss_fn = BCEWithLogitsLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-4)

    # get best loss
    best_loss = get_best_loss()

    # start training of model
    n_epochs = 2
    train_model(n_epochs, model, train_loader, device, optimizer, loss_fn, val_loader, best_loss)



    # f1 score + AUROC, mit tensorboard
    # https://pytorch.org/docs/stable/tensorboard.html