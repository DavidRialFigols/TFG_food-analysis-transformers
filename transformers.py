import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

import random
import time
from datetime import datetime

import os
import copy

import pandas as pd

import timm
from timm.optim import create_optimizer_v2
import torch.nn.functional as F

from collections import OrderedDict

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--transformer', help='Transformer to use', required=True, type=str)
parser.add_argument('-g', '--gpu', help='GPU to use', required=True, type=str)
parser.add_argument('-l', '--load_pretrain', help='directory with model', type=str)
parser.add_argument('-e', '--start_epoch', help='last epoch of the pretrained model', type=int)
parser.add_argument('-d', '--dynamic_block', default=False, action='store_true')

random.seed(42)
num_classes = 101
using_dataset = f"food-{num_classes}"
using_transformer = parser.parse_args().transformer
using_gpu = parser.parse_args().gpu
load_pretrain = parser.parse_args().load_pretrain
start_epoch = parser.parse_args().start_epoch
dynamic_block = parser.parse_args().dynamic_block

models_transformers = {
    'ViT': 'vit_base_patch16_224_in21k',
    'CSWin': 'CSWin_96_24322_base_224',
    'BEiT': 'beit_base_patch16_224_in22k',
    'SWin': 'swin_base_patch4_window7_224_in22k',
    'DeiT': 'deit_base_distilled_patch16_224',
}

num_blocks = {
    'ViT': 12,
    'BEiT': 12,
    'DeiT': 12,
    'SWin': {0: 2, 1: 2, 2: 18, 3: 2},
    'CSWin': {1: 2, 2: 4, 3: 32, 4: 2},
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

BATCH_SIZE = 32
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
device = torch.device(f"cuda:{using_gpu}")

if using_transformer == 'CSWin':
    model = load_checkpoint(model, 'cswin_base_224.pth')
#     checkpoint = torch.load('upernet_cswin_base.pth')
#     print(checkpoint)
#     model.load_state_dict(checkpoint['state_dict'])

previous_time = 0
if load_pretrain and start_epoch:
    print('Loading checkpoint')
    PATH = load_pretrain
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    last_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    acc = checkpoint['acc']
    previous_time = checkpoint['time']
    for state in opt.state_dict()['state'].values():
        for k,v in state.items():
            state[k] = v.cuda(int(using_gpu))
else:
    start_epoch = 0

model = model.to(device)

output_dir = f"./checkpoints/{using_dataset}/{using_transformer}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
os.system(f"mkdir {output_dir}")
# output_dir = "./checkpoints/food-101/DeiT/20211230-101645"

EPOCHS = 50
df_val = pd.DataFrame()
df_train = pd.DataFrame()

# previous_loss = 100
# matrices_to_block = ['blocks.1.attn.qkv.weight', 'blocks.2.attn.qkv.weight']
# model.train()
# for name, param in model.named_parameters():
#     if name in matrices_to_block:
#         param.requires_grad = False
#         print(name, param)

if using_transformer in ['ViT', 'BEiT', 'DeiT']:
    blocked = [{
        'qkv': 0,
        'mlp1': 0,
        'mlp2': 0,
        'w0': 0,
    } for block in range(num_blocks[using_transformer])]
elif using_transformer in ['CSWin', 'SWin']:
    blocked = []
    for layer in range(4):
        if using_transformer == 'CSWin':
            layer += 1
        blocked.append([{
                        'qkv': 0,
                        'mlp1': 0,
                        'mlp2': 0,
                        'w0': 0,
                        } for block in range(num_blocks[using_transformer][layer])])
blocked_matrices = []
print('TRAINING STARTED')
start_time = time.time()
for epoch in range(start_epoch, EPOCHS):
    last_model_dict = copy.deepcopy(model.state_dict())
    losses_test, accs_test = [], []
    model.train()
    for name, param in model.named_parameters():
        if name in blocked_matrices and param.requires_grad:
            print(f"La matriu {name} s'esta modificant i hauria d'estar bloquejada!")
    
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

    if dynamic_block:
        if using_transformer in ['ViT', 'BEiT', 'DeiT']:
            for block in range(num_blocks[using_transformer]):
                qkv_diff = abs(torch.sum(torch.diff(last_model_dict[f'blocks.{block}.attn.qkv.weight']-model.state_dict()[f'blocks.{block}.attn.qkv.weight'])).item())
                mlp1_diff = abs(torch.sum(torch.diff(last_model_dict[f'blocks.{block}.mlp.fc1.weight']-model.state_dict()[f'blocks.{block}.mlp.fc1.weight'])).item())
                mlp2_diff = abs(torch.sum(torch.diff(last_model_dict[f'blocks.{block}.mlp.fc2.weight']-model.state_dict()[f'blocks.{block}.mlp.fc2.weight'])).item())
                w0_diff = abs(torch.sum(torch.diff(last_model_dict[f'blocks.{block}.attn.proj.weight']-model.state_dict()[f'blocks.{block}.attn.proj.weight'])).item())
                print(f"block {block} | qkv: {qkv_diff} | mlp1: {mlp1_diff} | mlp2: {mlp2_diff} | w: {w0_diff}")
                if qkv_diff < 0.005 and blocked[block]['qkv'] < 2:
                    blocked[block]['qkv'] += 1
                    if blocked[block]['qkv'] == 2:
                        print(f"QKV block {block} blocked in epoch {epoch}")
                        for name, param in model.named_parameters():
                            if name == f"blocks.{block}.attn.qkv.weight":
                                param.requires_grad = False
                                blocked_matrices.append(name)
                elif blocked[block]['qkv'] < 2:
                    blocked[block]['qkv'] = 0
                if mlp1_diff < 0.005 and blocked[block]['mlp1'] < 2:
                    blocked[block]['mlp1'] += 1
                    if blocked[block]['mlp1'] == 2:
                        print(f"MLP1 block {block} blocked in epoch {epoch}")
                        for name, param in model.named_parameters():
                            if name == f"blocks.{block}.mlp.fc1.weight":
                                param.requires_grad = False
                                blocked_matrices.append(name)
                elif blocked[block]['mlp1'] < 2:
                    blocked[block]['mlp1'] = 0
                if mlp2_diff < 0.005 and blocked[block]['mlp2'] < 2:
                    blocked[block]['mlp2'] += 1
                    if blocked[block]['mlp2'] == 2:
                        print(f"MLP2 block {block} blocked in epoch {epoch}")
                        for name, param in model.named_parameters():
                            if name == f"blocks.{block}.mlp.fc2.weight":
                                param.requires_grad = False
                                blocked_matrices.append(name)
                elif blocked[block]['mlp2'] < 2:
                    blocked[block]['mlp2'] = 0
                if w0_diff < 0.005 and blocked[block]['w0'] < 2:
                    blocked[block]['w0'] += 1
                    if blocked[block]['w0'] == 2:
                        print(f"W0 block {block} blocked in epoch {epoch}")
                        for name, param in model.named_parameters():
                            if name == f"blocks.{block}.attn.proj.weight":
                                param.requires_grad = False
                                blocked_matrices.append(name)
                elif blocked[block]['w0'] < 2:
                    blocked[block]['w0'] = 0


    df_val[f"{using_transformer}_epoch_{epoch}_pred"] = val_pred
    df_val[f"{using_transformer}_epoch_{epoch}_labels"] = val_labels
    df_val.to_csv(f"results/{using_transformer}_epoch_{epoch}_val.csv")
    loss = torch.stack(losses_test).sum() / len(dl_test.dataset)
    acc = torch.stack(accs_test).sum() / len(dl_test.dataset)
    
    try:
        os.system(f"rm results/{using_transformer}_epoch_{epoch-2}_train.csv")
        os.system(f"rm results/{using_transformer}_epoch_{epoch-2}_val.csv")
    except:
        continue
    acc_time = time.time() - start_time + previous_time
    print(f'Epoch: {epoch+1:>2}    Loss: {loss.item():.3f}    Accuracy: {acc:.3f}    Acu.Time: {acc_time:.3f}')
    # if previous_loss >= loss.item():
    #     previous_loss = loss.item()
    # elif previous_loss < loss.item()-0.5:
    #     break 
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss': loss.item(),
        'acc': acc,
        'time': acc_time
    }, f"{output_dir}/model_{epoch}.pt")
