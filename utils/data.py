import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance



def cv_random_flip(img, label, depth, pdepth):
    if random.randint(0, 1) == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        pdepth = pdepth.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, depth, pdepth


def randomCrop(image, label, depth, pdepth):
    border = 30
    image_width, image_height = image.size
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1,
        (image_height - crop_win_height) >> 1,
        (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1
    )
    return (image.crop(random_region),
            label.crop(random_region),
            depth.crop(random_region),
            pdepth.crop(random_region))


def randomRotation(image, label, depth, pdepth):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
        pdepth = pdepth.rotate(random_angle, mode)
    return image, label, depth, pdepth


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        img[randX, randY] = 0 if random.randint(0, 1) == 0 else 255
    return Image.fromarray(img)


def _collect_by_stem(root, exts=(".jpg", ".png", ".bmp")):
    d = {}
    for f in os.listdir(root):
        lf = f.lower()
        if lf.endswith(exts):
            stem = os.path.splitext(f)[0]
            d[stem] = os.path.join(root, f)
    return d


# -------------------------
# dataset for training
# -------------------------
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, depth_root, pdepth_root, trainsize):
        self.trainsize = trainsize

        img_map = _collect_by_stem(image_root, exts=(".jpg", ".png"))
        gt_map = _collect_by_stem(gt_root, exts=(".jpg", ".png"))
        dep_map = _collect_by_stem(depth_root, exts=(".jpg", ".png", ".bmp"))
        pdep_map = _collect_by_stem(pdepth_root, exts=(".png", ".jpg", ".bmp"))

        common = sorted(list(set(img_map.keys()) & set(gt_map.keys()) & set(dep_map.keys()) & set(pdep_map.keys())))
        if len(common) == 0:
            raise RuntimeError("No matched samples across RGB/GT/Depth/PseudoDepth. Please check filenames(stems).")

        self.images = [img_map[k] for k in common]
        self.gts = [gt_map[k] for k in common]
        self.depths = [dep_map[k] for k in common]
        self.pdepths = [pdep_map[k] for k in common]
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])
        self.depth_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])
        self.pdepth_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth = self.binary_loader(self.depths[index])
        pdepth = self.binary_loader(self.pdepths[index])  

        image, gt, depth, pdepth = cv_random_flip(image, gt, depth, pdepth)
        image, gt, depth, pdepth = randomCrop(image, gt, depth, pdepth)
        image, gt, depth, pdepth = randomRotation(image, gt, depth, pdepth)

       
        image = colorEnhance(image)

       
        gt = randomPeper(gt)

        # to tensor
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth = self.depth_transform(depth)
        pdepth = self.pdepth_transform(pdepth)

        
        return image, gt, depth, pdepth

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, depth_root, pdepth_root,
               batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=False):
    dataset = SalObjDataset(image_root, gt_root, depth_root, pdepth_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


# -------------------------
# test dataset and loader
# -------------------------
class test_dataset:
    def __init__(self, image_root, gt_root, depth_root, pdepth_root, testsize):
        self.testsize = testsize

        img_map = _collect_by_stem(image_root, exts=(".jpg", ".png"))
        gt_map = _collect_by_stem(gt_root, exts=(".jpg", ".png"))
        dep_map = _collect_by_stem(depth_root, exts=(".jpg", ".png", ".bmp"))
        pdep_map = _collect_by_stem(pdepth_root, exts=(".png", ".jpg", ".bmp"))

        common = sorted(list(set(img_map.keys()) & set(gt_map.keys()) & set(dep_map.keys()) & set(pdep_map.keys())))
        if len(common) == 0:
            raise RuntimeError("No matched val samples across RGB/GT/Depth/PseudoDepth. Check stems.")

        self.images = [img_map[k] for k in common]
        self.gts = [gt_map[k] for k in common]
        self.depths = [dep_map[k] for k in common]
        self.pdepths = [pdep_map[k] for k in common]

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.depth_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()
        ])
        self.pdepth_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()
        ])

        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image_pil = self.rgb_loader(self.images[self.index])
        image = self.transform(image_pil).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])

        depth = self.binary_loader(self.depths[self.index])
        depth = self.depth_transform(depth).unsqueeze(0)

        pdepth = self.binary_loader(self.pdepths[self.index])
        pdepth = self.pdepth_transform(pdepth).unsqueeze(0)

        name = os.path.basename(self.images[self.index])
        image_for_post = image_pil.resize(gt.size)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index = (self.index + 1) % self.size
        return image, gt, depth, pdepth, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size