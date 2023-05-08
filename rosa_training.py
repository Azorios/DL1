import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.modules.loss import BCEWithLogitsLoss
from torchvision import models, datasets
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, ToTensor, Resize, CenterCrop
from torchvision.models import ResNet50_Weights
from datasets import load_dataset, concatenate_datasets
from transformers import AutoImageProcessor
import matplotlib.pyplot as plt
import numpy as np
from models.resnet50 import ResNet50
import lightning.pytorch as pl
#from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import ModelCheckpoint

if __name__ == '__main__':
    # used device for computing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device, 'will be used.')

    # datasets
    fake = load_dataset('poloclub/diffusiondb', '2m_random_10k', split='train', data_dir='./')
    real = load_dataset('frgfm/imagenette', '320px', split='train+validation', data_dir='./')

    # remove unnecessary comlumns
    fake = fake.remove_columns(
        ['prompt', 'seed', 'step', 'cfg', 'sampler', 'width', 'height', 'user_name', 'timestamp', 'image_nsfw',
         'prompt_nsfw'])
    real = real.remove_columns('label')

    # add label column
    fake = fake.map(lambda x: {'image': x['image'], 'label': 1})
    real = real.map(lambda x: {'image': x['image'], 'label': 0})

    # split fake dataset into train, validation and test sets
    fake_train_testvalid = fake.train_test_split(test_size=0.2)
    fake_test_valid = fake_train_testvalid['test'].train_test_split(test_size=0.5)

    # split real dataset into train, validation and test sets
    real_train_testvalid = real.train_test_split(test_size=0.2)
    real_test_valid = real_train_testvalid['test'].train_test_split(test_size=0.5)

    # combine fake and real datasets into single dataset for each split
    train_dataset = concatenate_datasets([fake_train_testvalid['train'], real_train_testvalid['train']])
    val_dataset = concatenate_datasets([fake_test_valid['train'], real_test_valid['train']])
    test_dataset = concatenate_datasets([fake_test_valid['test'], real_test_valid['test']])

    print(train_dataset)
    # transform/preprocess input
    train_transforms = Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    val_transforms = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transforms = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


    def transform_train(batch):
        batch['pixel_values'] = [train_transforms(img.convert("RGB")) for img in batch['image']]
        del batch['image']
        return batch


    def transform_val(batch):
        batch['pixel_values'] = [val_transforms(img.convert("RGB")) for img in batch['image']]
        del batch['image']
        return batch


    def transform_test(batch):
        batch['pixel_values'] = [test_transforms(img.convert("RGB")) for img in batch['image']]
        del batch['image']
        return batch


    # apply transforms and preprocessing
    train_dataset.set_transform(transform_train)
    val_dataset.set_transform(transform_val)
    test_dataset.set_transform(transform_test)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # show example image
    def inv_trans(image):
        '''
        Inverse transformations so that colors are in normal range.
        '''
        inverse_trans = Compose([
            Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])

        return inverse_trans(image)


    def imshow(images, labels, batch_size):
        '''
        Display images (batch) in one figure with labels.
        '''
        fig = plt.figure(figsize=(8, 5))
        rows, columns = 2, 8

        for i in range(batch_size):
            unnorm_im = inv_trans(images[i]).cpu().numpy()
            if int(labels[i]) == 0:
                label = 'real'
            else:
                label = 'generated'

            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(np.transpose(unnorm_im, (1, 2, 0)))
            plt.axis('off')
            plt.title(label)

        fig.tight_layout()
        plt.show()


    # showcase some training images
    data = next(iter(train_loader))
    images = data['pixel_values'].to(device)
    labels = data['label'].to(device)
    imshow(images, labels, batch_size)

    # load pretrained resnet50 model from PyTorch
    model = ResNet50()
    print(model)

    try:
        load_model = torch.jit.load('./models/resnet50_ds1.pth')
        with torch.no_grad():
            print(load_model)
        # model.load_state_dict(torch.load('./models/resnet50.pth'))
        # model.name = 'resnet50'
    except ValueError:
        pass

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,

        every_n_epochs=1)

    trainer = pl.Trainer(max_epochs=1, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_callback.best_model_path)
    trainer.test(model, test_loader)

    with torch.no_grad():
        traced = torch.jit.trace(checkpoint_callback.best_model_path, torch.rand(1, 3, 224, 224))
    torch.jit.save(traced, './models/resnet50_ds1.pth')

    reloaded = torch.jit.load('./models/resnet50_ds1.pth')
    with torch.no_grad():
        print(reloaded)

    # f1 score + AUROC, mit tensorboard
