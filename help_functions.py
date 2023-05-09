from torchvision.transforms import Normalize, Compose
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from sklearn.metrics import confusion_matrix, classification_report


def iterate_dataloader(dataloader, device):
    """
    Iterate through given dataloader and return image and label data.
    """
    data = next(iter(dataloader))
    images = data['pixel_values'].to(device)
    labels = data['label'].to(device)

    return images, labels


def inv_trans(image):
    """
    Inverse transformations so that colors are in normal range.
    """
    inverse_trans = Compose([
        Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])

    return inverse_trans(image)


def imshow(images, labels):
    """
    Display images (batch) in one figure with labels.
    """
    fig = plt.figure(figsize=(8, 5))
    rows, columns = 2, 8

    for i in range(16):
        unnormalize_img = inv_trans(images[i]).cpu().numpy()
        if int(labels[i]) == 0: label = 'real'
        else: label = 'generated'

        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(np.transpose(unnormalize_img, (1, 2, 0)))
        plt.axis('off')
        plt.title(label)

    fig.tight_layout()
    plt.show()


def get_best_loss():
    """
    Get the best loss from previous runs if available.
    """
    try:
        with open('./model_data/best_loss_ds1.pkl', 'rb') as file:
            best_loss = pickle.load(file)
        print(f'\nbest loss: {best_loss}')
    except FileNotFoundError:
        best_loss = 42

    return best_loss


def get_image_label(data, device):
    """
    Get image and label data. Convert label data to the same nn output shape.
    """
    image = data['pixel_values'].to(device)
    label = data['label'].unsqueeze(1).float().to(device)

    return image, label


def accuracy():
    acc=1
    return acc


def save_model_with_best_loss(model, val_loss, device):
    """
    Saves best model and best loss. Returns the best loss.
    """
    with torch.no_grad():
        traced = torch.jit.trace(model, torch.rand(1, 3, 224, 224).to(device))
    torch.jit.save(traced, './model_data/resnet50_ds1.pth')

    # save the best loss for next run
    with open('model_data/best_loss_ds1.pkl', 'wb') as file: pickle.dump(val_loss, file)
    print('model updated')

    return val_loss


def plot_loss(epoch_train_losses, epoch_test_losses, epochs):
    epochs = list(range(1, epochs+1))
    plt.plot(epochs, epoch_train_losses, 'r', label='Train Loss')
    plt.plot(epochs, epoch_test_losses, 'g', label='Validation Loss')
    plt.title('Train and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_acc(total_train_acc, total_val_acc, epochs):
    epochs = list(range(1, epochs+1))
    plt.plot(epochs, total_train_acc, 'r', label='Train Accuracy')
    plt.plot(epochs, total_val_acc, 'g', label='Validation Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def predict_all(test_loader, model, device):
    predictions, targets = [], []

    model.eval()
    for data in test_loader:
        image, label = get_image_label(data, device)

        output = torch.sigmoid(model(image))
        preds = torch.round(output)

        preds = preds.detach().cpu().numpy()
        label = label.detach().cpu().numpy()

        for i in range(len(preds)):
            predictions.append(preds[i])
            targets.append((label[i]))

    predictions = np.array(predictions)
    targets = np.array(targets)

    conf_mat = confusion_matrix(targets, predictions)
    class_rep = classification_report(targets, predictions, target_names=('Real', 'Generated'))

    print(conf_mat)
    print(class_rep)

    file = open(f'./model_data/resnet50_ds1_metrics.txt', 'w')
    file.write(f'ResNet50 Dataset1\n\nConfusion Matrix\n\n{conf_mat}\n\nClassification Report\n\n{class_rep}\n')
    file.close()
