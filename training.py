from tqdm import tqdm
import torch
from help_functions import get_image_label, accuracy, save_model_with_best_loss

'''
Training and Validation of model. Model is saved when validation loss decreases. 
If model has not improved in a while, stop training early.
'''


def train_model(n_epochs, model, train_loader, device, optimizer, loss_fn, val_loader, best_loss):
    total_train_losses = []
    total_val_losses = []
    total_train_acc = []
    total_val_acc = []
    early_stopping_counter = 0
    early_stopping_tolerance = 5

    for epoch in range(n_epochs):
        # enter training mode
        model.train()
        print('Starting training...')

        train_loss, total, correct = 0, 0, 0

        for _, data in enumerate(tqdm(train_loader)):  # iterate over batches
            image, label = get_image_label(data, device)

            # zero gradient
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # make prediction by giving model image
                output = model(image)
                probabilities = torch.sigmoid(output)

                # compute loss
                loss = loss_fn(output, label)

            train_loss += loss / len(train_loader)

            # calculate accuracy for training
            predictions = probabilities > 0.5
            correct += (predictions == label).sum().item()
            total += predictions.size(0)

            # perform backpropagation, update weights on model
            loss.backward()
            optimizer.step()

        total_train_losses.append(train_loss)
        print('\nEpoch : {}, train loss : {}'.format(epoch + 1, train_loss))

        # calculate training accuracy
        train_acc = float(correct) / float(total) * 100
        total_train_acc.append(train_acc)
        print("Got {} / {} with training accuracy {}".format(correct, total, train_acc))

        # validation doesnt requires gradient
        with torch.no_grad():
            # model to eval mode
            model.eval()

            val_loss, total, correct = 0, 0, 0

            for data in val_loader:
                image, label = get_image_label(data, device)

                # give model image to receive prediction
                output = model(image)
                probabilities = torch.sigmoid(output)

                # calculate validation loss
                v_loss = loss_fn(output, label)
                val_loss += v_loss / len(val_loader)

                # calculate accuracy for validation
                predictions = probabilities > 0.5
                correct += (predictions == label).sum().item()
                total += predictions.size(0)

            total_val_losses.append(val_loss)
            print('Epoch : {}, val loss : {}'.format(epoch + 1, val_loss))

            # calculate validation accuracy
            val_acc = float(correct) / float(total) * 100
            total_val_acc.append(val_acc)
            print("Got {} / {} with validation accuracy {}".format(correct, total, val_acc))

            print('val_loss: ', val_loss)
            print('best_loss ', best_loss)

            # save best model
            if val_loss <= best_loss: best_loss = save_model_with_best_loss(model, val_loss, device)

            # early stopping
            if val_loss > best_loss:
                early_stopping_counter += 1
                print(f'early stopping counter {early_stopping_counter} / {early_stopping_tolerance}')

            if early_stopping_counter == early_stopping_tolerance:
                print(f'early stopping counter {early_stopping_counter} / {early_stopping_tolerance}')
                print("Terminating: early stopping")
                break  # terminate training

    return total_train_losses, total_val_losses, total_train_acc, total_val_acc
