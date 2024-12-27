from torch.utils.data import Dataset
import torch
import os
import numpy as np
import random
from scipy import ndimage
from scipy.ndimage.interpolation import zoom

# ===== normalize over the dataset 
import torch
import os
import numpy as np
import random
from scipy import ndimage
from scipy.ndimage import zoom
from torchvision.transforms import ColorJitter


# ===== normalize over the dataset
def dataset_normalized(imgs):
    imgs = imgs
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs - imgs_mean) / imgs_std
    return imgs_normalized


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def add_gaussian_noise(image, mean=0, std=0.01):
    noise = np.random.normal(mean, std, image.shape)
    image = image + noise
    return np.clip(image, 0, 1)


def random_zoom_crop(image, label, zoom_factor=(0.8, 1.2)):
    zoom_factor = random.uniform(zoom_factor[0], zoom_factor[1])
    if zoom_factor == 1:
        return image, label
    h, w = image.shape[:2]
    zh, zw = int(h * zoom_factor), int(w * zoom_factor)
    image = zoom(image, (zoom_factor, zoom_factor, 1), order=3)
    label = zoom(label, (zoom_factor, zoom_factor, 1), order=0)
    if zoom_factor > 1:
        crop_h, crop_w = (zh - h) // 2, (zw - w) // 2
        image = image[crop_h:crop_h + h, crop_w:crop_w + w]
        label = label[crop_h:crop_h + h, crop_w:crop_w + w]
    else:
        pad_h, pad_w = (h - zh) // 2, (w - zw) // 2
        image = np.pad(image, ((pad_h, h - zh - pad_h), (pad_w, w - zw - pad_w), (0, 0)), mode='constant')
        label = np.pad(label, ((pad_h, h - zh - pad_h), (pad_w, w - zw - pad_w), (0, 0)), mode='constant')
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size
        self.color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

    def __call__(self, ori_img, mask):
        image, label = ori_img, mask

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        elif random.random() > 0.5:
            image = self.color_jitter(torch.tensor(image).permute(2, 0, 1)).permute(1, 2, 0).numpy()
        elif random.random() > 0.5:
            image, label = random_zoom_crop(image, label)
        elif random.random() > 0.5:
            image = add_gaussian_noise(image)

        x, y = label.shape[0:2]
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
        return image, label


class isic_loader(Dataset):
    def __init__(self, path_Data, mode, image_size=224):
        super(isic_loader, self).__init__()
        self.mode = mode
        data_path = os.path.join(path_Data, f'data_{self.mode}_li.npy')
        mask_path = os.path.join(path_Data, f'mask_{self.mode}_li.npy')
        print("data_path:", data_path)
        print("mask_path:", mask_path)
        self.data = np.load(data_path)
        self.mask = np.load(mask_path)

        self.data = dataset_normalized(self.data)
        self.mask = np.expand_dims(self.mask, axis=3)

        self.transform = RandomGenerator(output_size=(image_size, image_size))

    def __getitem__(self, indx):
        img = self.data[indx]
        seg = self.mask[indx]
        if self.mode == "train":
            img, seg = self.transform(img, seg)

        img = torch.tensor(img.copy()).float()
        seg = torch.tensor(seg.copy()).float()
        img = img.permute(2, 0, 1)
        seg = seg.permute(2, 0, 1)

        return {'image': img, 'mask': seg}

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    # from PIL import Image
    # img_pil_112 = Image.open('PSAX_US/Split_Label/train/CR3dca6a0-CR3dca87f-000030_31.png')
    # img_np_112 = np.array(img_pil_112)
    # img_pil_224 = img_pil_112.resize((224,224),Image.NEAREST)
    # img_np_224 = np.array(img_pil_224)
    # cur_pixel_num = 0
    # for i in range(256):
    #     pixel = np.sum(img_np_224 == i) 
    #     if pixel != 0 :
    #         print(f"i:{i} pixel:{pixel}")
    
    path_Data = "dataprocessing/"
    # data   = np.load(path_Data+'data_train.npy')
    mask   = np.load(path_Data+'mask_train.npy')
    for img_idx in range(mask.shape[0]):
       print(f"{img_idx}/{mask.shape[0]} have been processed!!!")
       pixel = np.sum(mask[img_idx] == 0) + np.sum(mask[img_idx] == 255)
       assert pixel == 224*224,f"img_idx:{img_idx}"