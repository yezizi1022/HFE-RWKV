import os
import shutil


# This script is used to create training, validation, and test datasets and organize them into separate folders for input.
def split_dataset(image_folder, train_txt, valid_txt, test_txt, output_folder):

    train_folder = os.path.join(output_folder, 'train')
    valid_folder = os.path.join(output_folder, 'valid')
    test_folder = os.path.join(output_folder, 'test')


    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(valid_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Read train.txt and copy the corresponding images to the train folder.
    with open(train_txt, 'r') as file:
        train_prefixes = [line.strip() for line in file]

    with open(valid_txt, 'r') as file:
        valid_prefixes = [line.strip() for line in file]

    with open(test_txt, 'r') as file:
        test_prefixes = [line.strip() for line in file]

    for file_name in os.listdir(image_folder):
        if file_name.endswith(".png"):
            prefix = os.path.splitext(file_name)[0]
            prefix = prefix.split('_')[0]
            source_path = os.path.join(image_folder, file_name)

            if prefix in train_prefixes:
                destination_path = os.path.join(train_folder, file_name)
                shutil.copy(source_path, destination_path)
            elif prefix in valid_prefixes:
                destination_path = os.path.join(valid_folder, file_name)
                shutil.copy(source_path, destination_path)
            elif prefix in test_prefixes:
                destination_path = os.path.join(test_folder, file_name)
                shutil.copy(source_path, destination_path)


if __name__ == "__main__":
    image_folder = '../PSAX_US/annotations' #'../PSAX_US/frames'

    train_txt = '../PSAX_US/train.txt'
    valid_txt = '../PSAX_US/val.txt'
    test_txt = '../PSAX_US/test.txt'

    output_folder = '../PSAX_US/Split_Label'

    split_dataset(image_folder, train_txt, valid_txt, test_txt, output_folder)


