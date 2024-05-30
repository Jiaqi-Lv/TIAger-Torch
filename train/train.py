import os

import numpy as np
import segmentation_models_pytorch as smp
import torch
from data import TestDataset, TiagerDataset, TrainDataset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


def train_one_epoch(model, training_loader, optimizer, loss_fn):
    running_loss = 0.0

    for i, data in enumerate(tqdm(training_loader, desc="train", leave=False)):
        imgs, masks = data["img"].cuda().float(), data["mask"].cuda().long()
        optimizer.zero_grad()

        outputs = model(imgs)
        loss = loss_fn(outputs, masks)
        loss.backward()

        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        # running_loss += loss.item()

    return running_loss / len(training_loader.sampler)


def train_model(
    model,
    train_loader,
    validation_loader,
    optimizer,
    loss_fn,
    save_dir,
    epochs,
    epoch_number=0,
):
    best_vloss = 100000000
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(epochs), desc="epochs", leave=True):
        print(f"EPOCH {epoch_number+1}")
        model.train()
        avg_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(
                tqdm(validation_loader, desc="validation", leave=False)
            ):
                vimgs, vmasks = vdata["img"].cuda().float(), vdata["mask"].cuda().long()
                voutputs = model(vimgs)
                vloss = loss_fn(voutputs, vmasks)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        train_losses.append(avg_loss)
        val_losses.append(avg_vloss)
        print(f"LOSS train {avg_loss} valid {avg_vloss}")

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_name = f"model_{epoch_number}.pth"
            model_path = os.path.join(save_dir, model_name)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

    train_loss_path = os.path.join(save_dir, "train_losses.txt")
    with open(train_loss_path, "w+") as fp:
        for item in train_losses:
            fp.write(f"{item}\n")
    test_loss_path = os.path.join(save_dir, "test_losses.txt")
    with open(test_loss_path, "w+") as fp:
        for item in val_losses:
            fp.write(f"{item}\n")
