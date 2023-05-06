import lightning.pytorch as pl
from torchvision import models
from torchvision.models import ResNet50_Weights
import torch.nn as nn
from torch.nn.modules.loss import BCEWithLogitsLoss
import torch
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC

class ResNet50(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        
        # load pretrained model
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # freeze parameters and add customizid final layer for binary classification
        for params in self.model.parameters(): params.requires_grad_ = False
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=1)

        self.loss_function = BCEWithLogitsLoss
        self.lr = lr

        self.acc = BinaryAccuracy()
        self.f1_score = BinaryF1Score()
        self.auroc = BinaryAUROC()

    def forward(self, images):
        return self.model(images)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.fc.parameters(), lr=self.lr)
    
    def training_step(self, batch, batchidx):
        images, labels = batch

        output = self.forward(images)
        predictions = torch.sigmoid(output)

        # compute loss
        loss = self.loss_function(output, labels)

        # calculate accuracy for training
        self.acc = (predictions, labels)

        self.log('train_loss', loss, on_epoch=True)
        self.log('train_acc', self.acc, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batchidx):
        images, labels = batch
        
        print(images)
        print(labels)

        output = self.forward(images)
        predictions = torch.sigmoid(output)

        # compute loss
        loss = self.loss_function(output, labels)

        # calculate accuracy for training
        self.acc = (predictions, labels)

        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', self.acc, on_epoch=True)

        return loss

    def test_step(self, batch, batchidx):
        images, labels = batch

        output = self.forward(images)
        predictions = torch.sigmoid(output)

        # calculate accuracy for training
        self.acc = (predictions, labels)
        self.f1_score = (predictions, labels)
        self.auroc = (predictions, labels)

        self.log('test_acc', self.acc, on_epoch=True)
        
