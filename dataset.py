from datasets import load_dataset, concatenate_datasets

'''
Download dataset, remove unnecessary columns and add label 0 to real images and 1 to generated images.
Then split the dataset into a train (80%), validation (10%) and test dataset (10%).
'''


def get_dataset():
    # datasets
    fake = load_dataset('poloclub/diffusiondb', '2m_random_10k', split='train', data_dir='./')
    real = load_dataset('frgfm/imagenette', '320px', split='train+validation', data_dir='./')

    # remove unnecessary columns
    fake = fake.remove_columns(
        ['prompt', 'seed', 'step', 'cfg', 'sampler', 'width', 'height', 'user_name', 'timestamp', 'image_nsfw',
         'prompt_nsfw'])
    real = real.remove_columns('label')

    # add label column with 0 for real images and for 1 for generated images
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

    return train_dataset, val_dataset, test_dataset
