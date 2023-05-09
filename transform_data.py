from torchvision.transforms import Normalize, Compose, RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, \
    ToTensor, Resize, CenterCrop

'''
Transform and preprocess data. Convert the images to pixel_values (tensors).
'''


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
