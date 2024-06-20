import os

import numpy as np
import segmentation_models_pytorch as smp
import torch
from data import TestDataset, TiagerDataset, TrainDataset
from loss_functions import BCE_Dice_Loss
from torch.utils.data import ConcatDataset, DataLoader, random_split

from train import train_model

BATCH_SIZE = 8

FOLD_DIR_1 = "/home/u1910100/Documents/Tiger_Data/tissue_segmentation/patches/256/fold_1"
FOLD_DIR_2 = "/home/u1910100/Documents/Tiger_Data/tissue_segmentation/patches/256/fold_2"
FOLD_DIR_3 = "/home/u1910100/Documents/Tiger_Data/tissue_segmentation/patches/256/fold_3"
FOLD_DIR_4 = "/home/u1910100/Documents/Tiger_Data/tissue_segmentation/patches/256/fold_4"
FOLD_DIR_5 = "/home/u1910100/Documents/Tiger_Data/tissue_segmentation/patches/256/fold_5"

test_fold = 1
SAVE_DIR = f"/home/u1910100/GitHub/TIAger-Torch/runs/tissue_fold_{test_fold}"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

dataset_1 = TiagerDataset(fold_dir=FOLD_DIR_1)
dataset_2 = TiagerDataset(fold_dir=FOLD_DIR_2)
dataset_3 = TiagerDataset(fold_dir=FOLD_DIR_3)
dataset_4 = TiagerDataset(fold_dir=FOLD_DIR_4)
dataset_5 = TiagerDataset(fold_dir=FOLD_DIR_5)

# subsetTrain, subsetTest = random_split(dataset, [0.7, 0.3])
subsetTrain = ConcatDataset([dataset_2, dataset_3, dataset_4, dataset_5])
subsetTest = dataset_1
train_dataset = TrainDataset(subsetTrain)
test_dataset = TestDataset(subsetTest)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True
)


model = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3,
)

model.to("cuda")

for param in model.encoder.parameters():
    param.requires_grad = False
# model.load_state_dict(torch.load('/home/u1910100/cloud_workspace/tissue_mask_model/runs/efficientunetb0/model_13.pth'))

loss_fn = smp.losses.JaccardLoss(mode="multiclass", from_logits=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.01)

train_model(
    model=model,
    train_loader=train_loader,
    validation_loader=validation_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    save_dir=SAVE_DIR,
    epochs=10,
    epoch_number=0,
)
