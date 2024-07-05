import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
import numpy as np
import shutil
from enum import Enum
import json
import os
from pathlib import Path
from PIL import Image
import yaml
import matplotlib.pyplot as plt
import cv2 as cv

MAP_ANIMAL3D_WNID = {
    "japanese_spaniel": "n02085782",
    "walker_hound": "n02089867",
    "redbone": "n02090379",
    "saluki": "n02091831",
    "weimaraner": "n02092339",
    "cairn": "n02096177",
    "boston_bull": "n02096585",
    "tibetan_terrier": "n02097474",
    "soft-coated_wheaten_terrier": "n02098105",
    "golden_retriever": "n02099601",
    "english_setter": "n02100583",
    "gordon_setter": "n02101006",
    "brittany_spaniel": "n02101388",
    "english_foxhound": "n02102040",
    "irish_terrier": "n02102973",
    "samoyed": "n02109525",
    "chow": "n02112137",
    "timber_wolf": "n02114367",
    "arctic_fox": "n02120079",
    "siberian_husky": "n02124075",
    "eskimo_dog": "n02109961",
    "borzoi": "n02128385",
    "beagle": "n02129604",
    "basset_hound": "n02130308",
    "brown_bear": "n02132136",
    "american_black_bear": "n02133161",
    "ice_bear": "n02134084",
    "sloth_bear": "n02134418",
    "hippopotamus": "n02389026",
    "zebra": "n02391049",
    "wild_boar": "n02397096",
    "yak": "n02403003",
    "water_buffalo": "n02408429",
    "ram": "n02412080",
    "bison": "n02415577",
    "ox": "n02417914",
    "hartebeest": "n02422106",
    "impala": "n02422699",
    "gazelle": "n02423022",
    "bighorn": "n02412080"
}

class ANIMAL3D_SUBCATEGORY(Enum):
    japanese_spaniel = 0
    walker_hound = 1
    redbone = 2
    saluki = 3
    weimaraner = 4
    cairn = 5
    boston_bull = 6
    tibetan_terrier = 7
    soft_coated_wheaten_terrier = 8
    golden_retriever = 9
    english_setter = 10
    gordon_setter = 11
    brittany_spaniel = 12
    english_foxhound = 13
    irish_terrier = 14
    samoyed = 15
    chow = 16
    timber_wolf = 17
    arctic_fox = 18
    siberian_husky = 19
    eskimo_dog = 20
    borzoi = 21
    beagle = 22
    basset_hound = 23
    brown_bear = 24
    american_black_bear = 25
    ice_bear = 26
    sloth_bear = 27
    hippopotamus = 28
    zebra = 29
    wild_boar = 30
    yak = 31
    water_buffalo = 32
    ram = 33
    bison = 34
    ox = 35
    hartebeest = 36
    impala = 37
    gazelle = 38
    bighorn = 39

class ANIMAL3D_SUPERCATEGORY(Enum):
    dog = 0
    horse = 1
    cow = 2
    hippo = 3
    tiger = 4


class Animal3DDataset(torch.utils.data.Dataset):
    def __init__(self, 
                    path,
                    split='train', 
                    category=None, 
                    resize_im=False, 
                    imsize=(224,224),
                    bbox_crop=False, 
                    training_batch=False, 
                    replications=1,
                    **kwargs):
        self.path = path
        self.split = split
        self.annotation_path = os.path.join(path, f"{split}.json")
        with open(self.annotation_path, 'r') as ann_file:
            self.annot_data = json.load(ann_file)
        self.imgs_path = os.path.join(path, "images", split)
        #self.pairs = self._generate_pair()
        if category is not None:
            self.pairs = [pair for pair in self.pairs if category in pair]


        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=mean, std=std)
        self.resize_im = resize_im
        self.resize = transforms.Resize(imsize, antialias=True)
        self.imsize = imsize
        self.bbox_crop = bbox_crop
        self.training_batch = training_batch



        self.cat_dict = self.annot_data['categories']
        self.super_cat_dict = self.annot_data['supercategories']
        self.n_cats = len(self.super_cat_dict)


            
        self.geom_augment = transforms.RandomResizedCrop(imsize, scale=(.5,1.5), interpolation=Image.BICUBIC, antialias=True)
        self.color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
 
    def __len__(self):
        return len(self.annot_data['data'])


    def __getitem__(self, idx):

        annotation = self.annot_data['data'][idx]
        

        category = annotation['category']
        super_category = annotation['supercategory']
        img_path = os.path.join(self.path , annotation['img_path'])
        mask_path = os.path.join(self.path , annotation['mask_path'])
        bbx = annotation['bbox']
        kps = torch.tensor(annotation['keypoint_2d'])
        kps_rescaled = kps / torch.tensor((annotation['width'],annotation['height'],1))
    
        kps = kps_rescaled * torch.tensor((*self.imsize,1))
        kps = kps.to(torch.int64)

        mask = Image.open(mask_path).convert('L')
        mask = self.to_tensor(mask)
        image= Image.open(img_path).convert('RGB')
        image_shape = image.size
        assert image_shape == (annotation['width'], annotation['height'])

        vp = self._getviewpoint(annotation['pose'][:3])

       # src_kps = torch.from_numpy(np.array(src_kps)).flip(-1)
       # trg_kps = torch.from_numpy(np.array(trg_kps)).flip(-1)

        #alpha = max(trg_bbx[2] - trg_bbx[0], trg_bbx[3] - trg_bbx[1])

        if self.training_batch:
            if self.bbox_crop:
                left = bbx[0]
                top = bbx[1]
                width = bbx[2] - bbx[0]
                height = bbx[3] - bbx[1]
                image = transforms.functional.crop(image, top, left, height, width)
                mask = transforms.functional.crop(mask, top, left, height, width)
            image = self.to_tensor(image)
            if self.resize_im:
                """
                src_im = self.to_tensor(src_im)
                combined = torch.cat([src_im, src_mask], dim=0)
                aug_combined = self.geom_augment(combined)
                src_im, src_mask = aug_combined[:3], aug_combined[3:]
                src_im = self.normalize(src_im)
                """
                image = self.to_tensor(image)
                combined = torch.cat([image, mask], dim=0)
                aug_combined = self.geom_augment(combined)
                image, mask = aug_combined[:3], aug_combined[3:]
                image = self.normalize(image)
            
            transforms.functional.to_tensor(mask)
            mask = self.resize(mask)
            return {'img': image, 'mask': mask, 'idx': idx, 'vp': vp, 'cat': category , 'super_cat': super_category , 'kps': kps}

    def _getviewpoint(self, axis_angle ,n_bins=8):
        x = torch.tensor(axis_angle[0])
        y = torch.tensor(axis_angle[1])
        azimuth = torch.arctan2(y, x)
        if azimuth < 0:
            azimuth += 2 * np.pi
        azimuth = azimuth * 180 / np.pi
        bin_size = 360 / n_bins
        azimuth_bin = torch.round(azimuth / bin_size)
        return azimuth_bin

if __name__ == '__main__':
    with open("configs/dataset/Animal3D.yaml") as config_file:
        cfg = yaml.safe_load(config_file)
    path = 'datasets/animal3d'
    dataset = Animal3DDataset
    train_dataset = dataset(path=cfg['data']['data_path'], resize_im=False, imsize=cfg['data']['im_size'], n_bins=cfg['data']['vp_bins'], training_batch=True, split='train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, num_workers=1, drop_last=False, shuffle=True,)
    for i, sample in enumerate(train_dataloader):
        print(f"img : {sample['img'].shape}")
        print(f"mask : {sample['mask'].shape}")
        print(f"idx : {sample['idx']}")
        print(f"vp : {sample['vp']}")
  
        if i <= 10:
            image_numpy = sample['img'][0].permute(1,2,0).numpy()
            image_numpy = np.ascontiguousarray(image_numpy)
            mask_numpy = sample['mask'][0].permute(1,2,0).numpy()
            #cv2_image = cv.cvtColor(image_numpy,)
            for kps in sample['kps'][0]:
                if kps[2] == 0:
                    continue
                kps = kps.numpy()

                cv.circle(image_numpy, (kps[0], kps[1]), 5, (0,255,0), -1)

            cv.imshow('mask', mask_numpy)
            cv.imshow('img', image_numpy)
            cv.waitKey(0)
            cv.destroyAllWindows()
            if i == 10:
                break
        #print(f"kps : {sample['kps'].shape}")
        
