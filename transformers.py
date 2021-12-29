import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

import random
import time
from datetime import datetime

import os

import pandas as pd

import timm
from timm.optim import create_optimizer_v2
import torch.nn.functional as F

from collections import OrderedDict

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--transformer', help='Transformer to use', required=True, type=str)

random.seed(42)
num_classes = 36
using_dataset = f"food-{num_classes}"
using_transformer = parser.parse_args().transformer

models_transformers = {
    'ViT': 'vit_base_patch16_224_in21k',
    'CSWin': 'CSWin_96_24322_base_224',
    'BEiT': 'beit_base_patch16_224_in22k',
    'SWin': 'swin_base_patch4_window7_224_in22k',
    'DeiT': 'deit_base_distilled_patch16_224',
}

data_transform_train = transforms.Compose(
            [transforms.Resize([256,256]),
             transforms.RandomCrop([224,224]),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ])
data_transform_test = transforms.Compose(
            [transforms.Resize([224,224]),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ])

ds_train = datasets.ImageFolder(f"{using_dataset}/train", transform=data_transform_train)
ds_test = datasets.ImageFolder(f"{using_dataset}/test", transform=data_transform_test)

BATCH_SIZE = 8
dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)
dl_train

def get_accuracy(predictions, labels):
    return (predictions.argmax(dim=1) == labels).float().mean()

def get_loss(predictions, labels):
    predictions = predictions.reshape(-1, predictions.shape[-1])
    #labels = labels.unsqueeze(1).expand(-1, 1).reshape(-1)
    return F.cross_entropy(predictions, labels)
def load_checkpoint(model, checkpoint_path):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        state_dict_key = 'state_dict_ema'
        model_key = 'model'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                if 'head' in k:
                    continue
                new_state_dict[name] = v
            state_dict = new_state_dict
    else:
        raise FileNotFoundError()
    model_dict = model.state_dict()
    pretrained_dict = state_dict
    loaded_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    load_dic = [k for k, v in pretrained_dict.items() if k in model_dict]
    miss_dic = [k for k, v in pretrained_dict.items() if not (k in model_dict)]
    unexpect_dic = [k for k, v in model_dict.items() if not (k in pretrained_dict)]
    model_dict.update(loaded_dict)
    model.load_state_dict(model_dict, strict=True)
    return model

model = timm.create_model(models_transformers[using_transformer], pretrained=True, num_classes=num_classes)
opt = create_optimizer_v2(model, lr=1e-3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if using_transformer == 'CSWin':
    old_dict = model.state_dict()
    print(old_dict)
    model = load_checkpoint(model, 'cswin_base_224.pth')
    new_dict = model.state_dict()
    print("==============================================")
    print(new_dict)
#     checkpoint = torch.load('upernet_cswin_base.pth')
#     print(checkpoint)
#     model.load_state_dict(checkpoint['state_dict'])

# PATH = f"./checkpoints/{using_dataset}/SWin/20211123-232319/model_18.pt"
# checkpoint = torch.load(PATH)
# model.load_state_dict(checkpoint['model_state_dict'])
# opt.load_state_dict(checkpoint['optimizer_state_dict'])
# last_epoch = checkpoint['epoch']
# loss = checkpoint['loss']
# acc = checkpoint['acc']

model = model.to(device)

EPOCHS = 20
df_val = pd.DataFrame()
df_train = pd.DataFrame()

previous_loss = 100
# matrices_to_block = ['blocks.1.attn.qkv.weight', 'blocks.2.attn.qkv.weight']
# model.train()
# for name, param in model.named_parameters():
#     if name in matrices_to_block:
#         param.requires_grad = False
#         print(name, param)


for epoch in range(EPOCHS):
    losses_test, accs_test = [], []
    model.train()
#     for name, param in model.named_parameters():
#         if name in matrices_to_block and param.requires_grad:
#             print(f"La matriu {name} s'esta modificant!")
    
    train_labels = []
    train_pred = []
    for images, labels in dl_train:
        opt.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        pred = model(images)
        if using_transformer == 'DeiT':
            pred = pred[0]
        loss = get_loss(pred, labels)
        acc = get_accuracy(pred, labels)
        train_pred.append(pred.cpu().detach().numpy())
        train_labels.append(labels.cpu().detach().numpy())
        loss.backward()
        opt.step()
    df_train[f"epoch_{epoch+1}_pred"] = train_pred
    df_train[f"epoch_{epoch+1}_labels"] = train_labels
    df_train.to_csv(f"results/{using_transformer}_epoch_{epoch+1}_train.csv")
    
    model.eval()
    val_labels = []
    val_pred = []
    with torch.no_grad():
        for images, labels in dl_test:
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            loss = get_loss(pred, labels)
            acc = get_accuracy(pred, labels)
            accs_test.append(acc * images.shape[0])            
            losses_test.append(loss * images.shape[0])
            val_labels.append(labels.cpu().detach().numpy())
            val_pred.append(pred.cpu().detach().numpy())
            
    df_val[f"{using_transformer}_epoch_{epoch}_pred"] = val_pred
    df_val[f"{using_transformer}_epoch_{epoch}_labels"] = val_labels
    loss = torch.stack(losses_test).sum() / len(dl_test.dataset)
    acc = torch.stack(accs_test).sum() / len(dl_test.dataset)
    
    print(f'Epoch: {epoch+1:>2}    Loss: {loss.item():.3f}    Accuracy: {acc:.3f}')
    # if previous_loss >= loss.item():
    #     previous_loss = loss.item()
    # elif previous_loss < loss.item()-0.5:
    #     break 
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss': loss.item(),
        'acc': acc
    }, f"{output_dir}/model_{epoch}.pt")