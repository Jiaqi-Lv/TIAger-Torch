import os

import albumentations as alb
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def augmentation(img, mask):
    aug = alb.Compose(
        [
            alb.OneOf(
                [
                    alb.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=(-10, 10),
                        val_shift_limit=0,
                        always_apply=False,
                        p=0.75,
                    ),  # .8
                    alb.RGBShift(
                        r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.75
                    ),  # .7
                ],
                p=1.0,
            ),
            alb.OneOf(
                [
                    alb.GaussianBlur(blur_limit=(3, 5), p=0.5),
                    alb.Sharpen(alpha=(0.1, 0.3), lightness=(1.0, 1.0), p=0.5),
                    alb.ImageCompression(quality_lower=30, quality_upper=80, p=0.5),
                ],
                p=1.0,
            ),
            alb.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, p=0.5
            ),
            alb.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=180,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5,
            ),
            alb.Flip(p=0.5),
        ],
        p=0.5,
    )
    transformed = aug(image=img, mask=mask)
    img, mask = transformed["image"], transformed["mask"]
    return img, mask


class TiagerDataset(Dataset):
    def __init__(self, fold_dir) -> None:
        self.fold_dir = fold_dir
        self.files = os.listdir(fold_dir)
        self.total = len(self.files)

    def __len__(self):
        return self.total

    def merge_labels(self, mask):
        mask[mask == 3] = 0
        mask[mask == 4] = 0
        mask[mask == 5] = 0
        mask[mask == 7] = 0
        mask[mask == 6] = 2
        return mask

    def __getitem__(self, index):
        file_path = os.path.join(self.fold_dir, self.files[index])
        img_file = np.load(file_path)
        img = img_file[:, :, 0:3]
        mask = img_file[:, :, 3]
        mask = self.merge_labels(mask)
        return img, mask


class TiagerCellsDataset(Dataset):
    def __init__(self, file_list) -> None:
        self.files = file_list
        self.total = len(self.files)

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        file_path = self.files[index]
        img_file = np.load(file_path)
        img = img_file[:, :, 0:3]
        mask = img_file[:, :, 3]
        return img, mask, file_path


class TrainDataset(Dataset):
    def __init__(self, subset, do_aug=True):
        self.subset = subset
        self.do_aug = do_aug

    def imagenet_normalise(self, img):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img - mean
        img = img / std
        return img

    def __getitem__(self, index):
        img, mask, file_path = self.subset[index]
        if self.do_aug:
            img, mask = augmentation(img, mask)

        mask = mask[:, :, np.newaxis]
        img = img / 255
        img = self.imagenet_normalise(img)

        img = np.moveaxis(img, 2, 0)
        mask = np.moveaxis(mask, 2, 0)
        return {"img": img, "mask": mask, "file_name": file_path}

    def __len__(self):
        return len(self.subset)


class TestDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def imagenet_normalise(self, img):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img - mean
        img = img / std
        return img

    def __getitem__(self, index):
        img, mask, file_path = self.subset[index]

        mask = mask[:, :, np.newaxis]
        img = img / 255
        img = self.imagenet_normalise(img)

        img = np.moveaxis(img, 2, 0)
        mask = np.moveaxis(mask, 2, 0)
        return {"img": img, "mask": mask, "file_name": file_path}

    def __len__(self):
        return len(self.subset)


def get_cell_dataloaders(patch_folder, fold_num, batch_size=32, phase="Train"):
    fold_folders = []
    for i in range(0, 5):
        fold_folder_path = os.path.join(patch_folder, f"fold_{i+1}/")
        fold_folders.append(fold_folder_path)

    test_fold_folder = fold_folders.pop(fold_num - 1)

    train_files = []
    test_files = []

    for file_name in os.listdir(test_fold_folder):
        full_path = os.path.join(test_fold_folder, file_name)
        test_files.append(full_path)

    for fold_folder_path in fold_folders:
        for file_name in os.listdir(fold_folder_path):
            full_path = os.path.join(fold_folder_path, file_name)
            train_files.append(full_path)

    print(f"Train files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")

    if phase == "Train":
        classes = []
        class_counts = [0, 0]  # negatives, positives
        for file in train_files:
            fn = os.path.splitext(file)[0]
            if fn[-1] == "p":
                classes.append(1)
                class_counts[1] += 1
            if fn[-1] == "n":
                classes.append(0)
                class_counts[0] += 1

        print(class_counts)
        sample_weights = []
        for i in classes:
            sample_weights.append(1 / class_counts[i])

        train_sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_files), replacement=True
        )

        train_set = TiagerCellsDataset(train_files)
        train_set = TrainDataset(train_set)
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=False, sampler=train_sampler
        )

        test_set = TiagerCellsDataset(test_files)
        test_set = TestDataset(test_set)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

        return train_loader, test_loader

    else:
        train_set = TiagerCellsDataset(train_files)
        train_set = TrainDataset(train_set, do_aug=False)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        test_set = TiagerCellsDataset(test_files)
        test_set = TestDataset(test_set)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

        return train_loader, test_loader


if __name__ == "__main__":
    patch_folder = (
        "/home/u1910100/Documents/Tiger_Data/cell_detection/dilation/patches/128/"
    )
    get_cell_dataloaders(patch_folder, 5)
