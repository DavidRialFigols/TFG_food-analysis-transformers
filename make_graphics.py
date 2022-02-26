import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

import random
import time
from datetime import datetime

import os

import pandas as pd
import numpy as np

import timm
from timm.optim import create_optimizer_v2
import torch.nn.functional as F

import seaborn as sns
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--transformer', help='Transformer to use', required=True, type=str)
parser.add_argument('-d', '--datetime', help='Datetime of the checkpoint', required=True, type=str)
parser.add_argument('-n', '--num_epochs', help='Number of epochs', required=True, type=int)

random.seed(42)
num_classes = 101
using_dataset = f"food-{num_classes}"
using_transformer = parser.parse_args().transformer
datetime_transformer = parser.parse_args().datetime
num_epochs = parser.parse_args().num_epochs

layers = {
    'CSWin': {
        '1': [2, 96],
        '2': [4, 192],
        '3': [32, 384],
        '4': [2, 768]
    },
    'SWin': {
        '0': [2, 128],
        '1': [2, 256],
        '2': [18, 512],
        '3': [2, 1024]
    }
}

fig_size = (10,8)

models_dict = []
accs = []
def load_model_checkpoint(epoch, datetime):
    PATH = f"./checkpoints/{using_dataset}/{using_transformer}/{datetime}/model_{epoch}.pt"
    checkpoint = torch.load(PATH, 'cpu')
    last_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    acc = checkpoint['acc']
    acc_time = checkpoint['time']
    print(f"epoch {epoch} | accuracy: {acc} | loss: {loss} | acc_time: {acc_time}")
    models_dict.append(checkpoint['model_state_dict'])
    accs.append(acc)
    del checkpoint
    torch.cuda.empty_cache()

for i in range(num_epochs):
    load_model_checkpoint(i, datetime_transformer)

atributes_model = {}
for model in models_dict:
    for i in model:
        if(i not in atributes_model):
            atributes_model[i] = []
        atributes_model[i].append(model[i])

def graphics_sv_epochs(models_dict, using_transformer, mat_size, num_blocks, fig_size, layer=False):
    for epoch, model in enumerate(models_dict): # the epoch is fixed in each graphic
    	if epoch%4==0:
	        q11 = []
	        k11 = []
	        v11 = []
	        q_svdvals = pd.DataFrame()
	        k_svdvals = pd.DataFrame()
	        v_svdvals = pd.DataFrame()
	        if using_transformer == "CSWin":
	            output = f"grafics/{using_transformer}_stage{layer}_epoch-{epoch}_singular-values_"
	        elif using_transformer == "SWin":
	            output = f"grafics/{using_transformer}_layer-{layer}_epoch-{epoch}_singular-values_"
	        else:
	            output = f"grafics/{using_transformer}_epoch-{epoch}_singular-values_"
	        for n in range(num_blocks):
	            if using_transformer == "CSWin":
	                i = f'stage{layer}.{n}.qkv.weight'
	            elif using_transformer == "SWin":
	                i = f'layers.{layer}.blocks.{n}.attn.qkv.weight'
	            else:
	                i = f'blocks.{n}.attn.qkv.weight'
	            for j in model:
	                if i==j:
	                    q11.append(model[j][:mat_size])
	                    k11.append(model[j][mat_size:mat_size*2])
	                    v11.append(model[j][mat_size*2:mat_size*3])
	            for l in range(len(q11)):
	                q_svdvals[f"block{l}"] = torch.linalg.svdvals(q11[l])
	                k_svdvals[f"block{l}"] = torch.linalg.svdvals(k11[l])
	                v_svdvals[f"block{l}"] = torch.linalg.svdvals(v11[l])
	        # make the heatmaps
	        q_svdvals = q_svdvals
	        k_svdvals = k_svdvals
	        v_svdvals = v_svdvals
	        fig, ax = plt.subplots(figsize=fig_size)
	        sns.scatterplot(data=q_svdvals.iloc[:200, :], s=15).set(title=f"{using_transformer} - Singular values of the Q matrix in the {epoch}th epoch", ylabel="Singular Value", xlabel = "Position of Singular Value")
	        plt.savefig(f"{output}q.png", dpi=750)
	        plt.close()


	        fig, ax = plt.subplots(figsize=fig_size)
	        sns.scatterplot(data=k_svdvals.iloc[:200, :], s=15).set(title=f"{using_transformer} - Singular values of the K matrix in the {epoch}th epoch", ylabel="Singular Value", xlabel = "Position of Singular Value")
	        plt.savefig(f"{output}k.png", dpi=750)
	        plt.close()


	        fig, ax = plt.subplots(figsize=fig_size)
	        sns.scatterplot(data=v_svdvals.iloc[:200, :], s=15).set(title=f"{using_transformer} - Singular values of the V matrix in the {epoch}th epoch", ylabel="Singular Value", xlabel = "Position of Singular Value")
	        plt.savefig(f"{output}v.png", dpi=750)
	        plt.close()

def graphics_sv_blocks(models_dict, using_transformer, mat_size, num_blocks, fig_size, layer=False):
    for n in range(num_blocks):
        q11 = []
        k11 = []
        v11 = []
        q_svdvals = pd.DataFrame()
        k_svdvals = pd.DataFrame()
        v_svdvals = pd.DataFrame()
        if using_transformer == "CSWin":
            i = f'stage{layer}.{n}.qkv.weight'
            output = f"grafics/{using_transformer}_stage{layer}_bloc-{n}_singular-values_"
        elif using_transformer == "SWin":
            i = f'layers.{layer}.blocks.{n}.attn.qkv.weight'
            output = f"grafics/{using_transformer}_layer-{layer}_bloc-{n}_singular-values_"
        else:
            i = f'blocks.{n}.attn.qkv.weight'
            output = f"grafics/{using_transformer}_bloc-{n}_singular-values_"
        for j in atributes_model[i]:
            q = j[:mat_size]
            k = j[mat_size:mat_size*2]
            v = j[mat_size*2:mat_size*3]
            q11.append(q)
            k11.append(k)
            v11.append(v)
        for l in range(len(q11)):
            q_svdvals[f"e{l}"] = torch.linalg.svdvals(q11[l])
            k_svdvals[f"e{l}"] = torch.linalg.svdvals(k11[l])
            v_svdvals[f"e{l}"] = torch.linalg.svdvals(v11[l])

        # plot the graphs
        q_svdvals = q_svdvals
        k_svdvals = k_svdvals
        v_svdvals = v_svdvals
        fig, ax = plt.subplots(figsize=fig_size)
        if q_svdvals.shape[1] > 12:
            indexs = [i for i in range(q_svdvals.shape[1]) if i%4==0]
        else:
            indexs = [i for i in range(q_svdvals.shape[1])]
        sns.scatterplot(data=q_svdvals.iloc[:200, indexs], s=15).set(title=f"{using_transformer} - Evolution of the singular values of the Q matrix in the block {n}", ylabel="Singular Value", xlabel = "Position of Singular Value")
        plt.savefig(f"{output}q.png", dpi=750)
        plt.close()

        fig, ax = plt.subplots(figsize=fig_size)
        if k_svdvals.shape[1] > 12:
            indexs = [i for i in range(k_svdvals.shape[1]) if i%4==0]
        else:
            indexs = [i for i in range(k_svdvals.shape[1])]
        sns.scatterplot(data=k_svdvals.iloc[:200, indexs], s=15).set(title=f"{using_transformer} - Evolution of the singular values of the K matrix in the block {n}", ylabel="Singular Value", xlabel = "Position of Singular Value")
        plt.savefig(f"{output}k.png", dpi=750)
        plt.close()

        fig, ax = plt.subplots(figsize=fig_size)
        if v_svdvals.shape[1] > 12:
            indexs = [i for i in range(v_svdvals.shape[1]) if i%4==0]
        else:
            indexs = [i for i in range(v_svdvals.shape[1])]
        sns.scatterplot(data=v_svdvals.iloc[:200, indexs], s=15).set(title=f"{using_transformer} - Evolution of the singular values of the V matrix in the block {n}", ylabel="Singular Value", xlabel = "Position of Singular Value")
        plt.savefig(f"{output}v.png", dpi=750)
        plt.close()
        #print(f"epoch {l} | q: {torch.linalg.svdvals(q11[l]).sum()} | k: {torch.linalg.svdvals(k11[l]).sum()} | v: {torch.linalg.svdvals(v11[l]).sum()}")

def graphics_ranks(models_dict, using_transformer, mat_size, num_blocks, fig_size, layer=False):
    q_ranks = {}
    k_ranks = {}
    v_ranks = {}
    for n in range(num_blocks):
        q11 = []
        k11 = []
        v11 = []
        q_ranks[f'block{n}'] = []
        k_ranks[f'block{n}'] = []
        v_ranks[f'block{n}'] = []
        if using_transformer == "CSWin":
            i = f'stage{layer}.{n}.qkv.weight'
            output = f"grafics/{using_transformer}_stage{layer}_rank_"
        elif using_transformer == "SWin":
            i = f'layers.{layer}.blocks.{n}.attn.qkv.weight'
            output = f"grafics/{using_transformer}_layer-{layer}_rank_"
        else:
            i = f'blocks.{n}.attn.qkv.weight'
            output = f"grafics/{using_transformer}_rank_"
        for j in atributes_model[i]:
            q = j[:mat_size]
            k = j[mat_size:mat_size*2]
            v = j[mat_size*2:mat_size*3]
            q11.append(q)
            k11.append(k)
            v11.append(v)
        for l in range(len(q11)):
            q_ranks[f"block{n}"].append(torch.linalg.matrix_rank(q11[l], tol=0.1).item())
            k_ranks[f"block{n}"].append(torch.linalg.matrix_rank(k11[l], tol=0.1).item())
            v_ranks[f"block{n}"].append(torch.linalg.matrix_rank(v11[l], tol=0.1).item())

    # plot the graphs
    q_ranks = pd.DataFrame(q_ranks)
    k_ranks = pd.DataFrame(k_ranks)
    v_ranks = pd.DataFrame(v_ranks)

    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(data=q_ranks.iloc[:200, :], markers=True, legend=False).set(title=f"{using_transformer} - Evolution of the ranks of the Q matrices", ylabel="Rank", xlabel = "Epoch")
    plt.legend(labels=q_ranks.columns, loc='upper right')
    plt.savefig(f"{output}q.png", dpi=750)
    plt.close()

    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(data=k_ranks.iloc[:200, :], markers=True, legend=False).set(title=f"{using_transformer} - Evolution of the ranks of the K matrices", ylabel="Rank", xlabel = "Epoch")
    plt.legend(labels=k_ranks.columns, loc='upper right')
    plt.savefig(f"{output}k.png", dpi=750)
    plt.close()

    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(data=v_ranks.iloc[:200, :], markers=True, legend=False).set(title=f"{using_transformer} - Evolution of the ranks of the V matrices", ylabel="Rank", xlabel = "Epoch")
    plt.legend(labels=v_ranks.columns, loc='upper right')
    plt.savefig(f"{output}v.png", dpi=750)
    plt.close()
    #print(f"epoch {l} | q: {torch.linalg.svdvals(q11[l]).sum()} | k: {torch.linalg.svdvals(k11[l]).sum()} | v: {torch.linalg.svdvals(v11[l]).sum()}")

def graphics_norms(models_dict, using_transformer, mat_size, num_blocks, fig_size, layer=False):
    q_norms = {}
    k_norms = {}
    v_norms = {}
    for n in range(num_blocks):
        q11 = []
        k11 = []
        v11 = []
        q_norms[f'block{n}'] = []
        k_norms[f'block{n}'] = []
        v_norms[f'block{n}'] = []
        if using_transformer == "CSWin":
            i = f'stage{layer}.{n}.qkv.weight'
            output = f"grafics/{using_transformer}_stage{layer}_norm_"
        elif using_transformer == "SWin":
            i = f'layers.{layer}.blocks.{n}.attn.qkv.weight'
            output = f"grafics/{using_transformer}_layer-{layer}_norm_"
        else:
            i = f'blocks.{n}.attn.qkv.weight'
            output = f"grafics/{using_transformer}_norm_"
        for j in atributes_model[i]:
            q = j[:mat_size]
            k = j[mat_size:mat_size*2]
            v = j[mat_size*2:mat_size*3]
            q11.append(q)
            k11.append(k)
            v11.append(v)
        for l in range(len(q11)):
            q_norms[f"block{n}"].append(torch.linalg.matrix_norm(q11[l]).item())
            k_norms[f"block{n}"].append(torch.linalg.matrix_norm(k11[l]).item())
            v_norms[f"block{n}"].append(torch.linalg.matrix_norm(v11[l]).item())

    # plot the graphs
    q_norms = pd.DataFrame(q_norms)
    k_norms = pd.DataFrame(k_norms)
    v_norms = pd.DataFrame(v_norms)
    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(data=q_norms.iloc[:200, :], markers=True, legend=False).set(title=f"{using_transformer} - Evolution of the Frobenius norms of the Q matrices", ylabel="Norm", xlabel = "Epoch")
    plt.legend(labels=q_norms.columns, loc='upper right')
    plt.savefig(f"{output}q.png", dpi=750)
    plt.close()

    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(data=k_norms.iloc[:200, :], markers=True, legend=False).set(title=f"{using_transformer} - Evolution of the Frobenius norms of the K matrices", ylabel="Norm", xlabel = "Epoch")
    plt.legend(labels=k_norms.columns, loc='upper right')
    plt.savefig(f"{output}k.png", dpi=750)
    plt.close()

    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(data=v_norms.iloc[:200, :], markers=True, legend=False).set(title=f"{using_transformer} - Evolution of the Frobenius norms of the V matrices", ylabel="Norm", xlabel = "Epoch")
    plt.legend(labels=v_norms.columns, loc='upper right')
    plt.savefig(f"{output}v.png", dpi=750)
    plt.close()
    #print(f"epoch {l} | q: {torch.linalg.svdvals(q11[l]).sum()} | k: {torch.linalg.svdvals(k11[l]).sum()} | v: {torch.linalg.svdvals(v11[l]).sum()}")

def graphics_diff_epochs(models_dict, using_transformer, mat_size, num_blocks, fig_size, layer=False):
    q_diffs = {}
    k_diffs = {}
    v_diffs = {}
    q_diffs_2 = {}
    k_diffs_2 = {}
    v_diffs_2 = {}
    for n in range(num_blocks):
        q11 = []
        k11 = []
        v11 = []
        q_diffs[f'block{n}'] = []
        k_diffs[f'block{n}'] = []
        v_diffs[f'block{n}'] = []
        q_diffs_2[f'block{n}'] = []
        k_diffs_2[f'block{n}'] = []
        v_diffs_2[f'block{n}'] = []
        if using_transformer == "CSWin":
            i = f'stage{layer}.{n}.qkv.weight'
            output = f"grafics/{using_transformer}_stage{layer}_diffs_"
        elif using_transformer == "SWin":
            i = f'layers.{layer}.blocks.{n}.attn.qkv.weight'
            output = f"grafics/{using_transformer}_layer-{layer}_diffs_"
        else:
            i = f'blocks.{n}.attn.qkv.weight'
            output = f"grafics/{using_transformer}_diffs_"
        for j in atributes_model[i]:
            q = j[:mat_size]
            k = j[mat_size:mat_size*2]
            v = j[mat_size*2:mat_size*3]
            q11.append(q)
            k11.append(k)
            v11.append(v)
        for l in range(1, len(q11)):
            q_diffs[f"block{n}"].append(torch.sum(torch.abs(q11[l]-q11[l-1])).item())
            k_diffs[f"block{n}"].append(torch.sum(torch.abs(k11[l]-k11[l-1])).item())
            v_diffs[f"block{n}"].append(torch.sum(torch.abs(v11[l]-v11[l-1])).item())
            q_diffs_2[f"block{n}"].append(mat_size*mat_size*max(0.05*abs(torch.sum(q11[l]-q11[l-1]).item())/(mat_size*mat_size), torch.std(q11[l]-q11[l-1], unbiased=False).item()))
            k_diffs_2[f"block{n}"].append(mat_size*mat_size*max(0.05*abs(torch.sum(k11[l]-k11[l-1]).item())/(mat_size*mat_size), torch.std(k11[l]-k11[l-1], unbiased=False).item()))
            v_diffs_2[f"block{n}"].append(mat_size*mat_size*max(0.05*abs(torch.sum(v11[l]-v11[l-1]).item())/(mat_size*mat_size), torch.std(v11[l]-v11[l-1], unbiased=False).item()))

    # plot the graphs
    q_diffs = pd.DataFrame(q_diffs)
    k_diffs = pd.DataFrame(k_diffs)
    v_diffs = pd.DataFrame(v_diffs)
    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(data=q_diffs.iloc[:200, :], markers=True, legend=False)
    plt.savefig(f"{output}q.png", dpi=750)
    plt.close()

    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(data=k_diffs.iloc[:200, :], markers=True, legend=False)
    plt.savefig(f"{output}k.png", dpi=750)
    plt.close()

    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(data=v_diffs.iloc[:200, :], markers=True, legend=False)
    plt.savefig(f"{output}v.png", dpi=750)
    plt.close()

    # plot the graphs
    q_diffs_2 = pd.DataFrame(q_diffs_2)
    k_diffs_2 = pd.DataFrame(k_diffs_2)
    v_diffs_2 = pd.DataFrame(v_diffs_2)
    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(data=q_diffs_2.iloc[:200, :], markers=True, legend=False)
    plt.savefig(f"{output}q_2.png", dpi=750)
    plt.close()

    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(data=k_diffs_2.iloc[:200, :], markers=True, legend=False)
    plt.savefig(f"{output}k_2.png", dpi=750)
    plt.close()

    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(data=v_diffs_2.iloc[:200, :], markers=True, legend=False)
    plt.savefig(f"{output}v_2.png", dpi=750)
    plt.close()

def graphics_sv_layers_epochs(models_dict, using_transformer, fig_size, layers):
    for epoch, model in enumerate(models_dict): # the epoch is fixed in each graphic
    	if epoch%4==0:
	        q_svdvals_means = pd.DataFrame()
	        k_svdvals_means = pd.DataFrame()
	        v_svdvals_means = pd.DataFrame()
	        for layer in range(4):
	            if using_transformer == 'CSWin':
	                layer += 1
	            layer = str(layer)
	            q11 = []
	            k11 = []
	            v11 = []
	            q_svdvals = pd.DataFrame()
	            k_svdvals = pd.DataFrame()
	            v_svdvals = pd.DataFrame()
	            num_blocks = layers[using_transformer][layer][0]
	            mat_size = layers[using_transformer][layer][1]
	            if using_transformer == "CSWin":
	                output = f"grafics/{using_transformer}_epoch-{epoch}_singular-values_"
	            elif using_transformer == "SWin":
	                output = f"grafics/{using_transformer}_epoch-{epoch}_singular-values_"
	            for n in range(num_blocks):
	                if using_transformer == "CSWin":
	                    i = f'stage{layer}.{n}.qkv.weight'
	                elif using_transformer == "SWin":
	                    i = f'layers.{layer}.blocks.{n}.attn.qkv.weight'
	                for j in model:
	                    if i==j:
	                        q11.append(model[j][:mat_size])
	                        k11.append(model[j][mat_size:mat_size*2])
	                        v11.append(model[j][mat_size*2:mat_size*3])
	                for l in range(len(q11)):
	                    q_svdvals[f"block{l}"] = torch.linalg.svdvals(q11[l])
	                    k_svdvals[f"block{l}"] = torch.linalg.svdvals(k11[l])
	                    v_svdvals[f"block{l}"] = torch.linalg.svdvals(v11[l])
	            if using_transformer == "CSWin":
	                q_svdvals_means[f"stage{layer}"] = q_svdvals.mean(axis=1)
	                k_svdvals_means[f"stage{layer}"] = k_svdvals.mean(axis=1)
	                v_svdvals_means[f"stage{layer}"] = v_svdvals.mean(axis=1) 
	            elif using_transformer == "SWin":
	                q_svdvals_means[f"layer{layer}"] = q_svdvals.mean(axis=1)
	                k_svdvals_means[f"layer{layer}"] = k_svdvals.mean(axis=1)
	                v_svdvals_means[f"layer{layer}"] = v_svdvals.mean(axis=1) 

	        # make the heatmaps
	        q_svdvals = q_svdvals_means
	        k_svdvals = k_svdvals_means
	        v_svdvals = v_svdvals_means
	        fig, ax = plt.subplots(figsize=fig_size)
	        sns.scatterplot(data=q_svdvals.iloc[:200, :], s=15).set(title=f"{using_transformer} - Singular values of the Q matrix in the {epoch}th epoch", ylabel="Singular Value", xlabel = "Position of Singular Value")
	        plt.savefig(f"{output}q.png", dpi=750)
	        plt.close()


	        fig, ax = plt.subplots(figsize=fig_size)
	        sns.scatterplot(data=k_svdvals.iloc[:200, :], s=15).set(title=f"{using_transformer} - Singular values of the K matrix in the {epoch}th epoch", ylabel="Singular Value", xlabel = "Position of Singular Value")
	        plt.savefig(f"{output}k.png", dpi=750)
	        plt.close()


	        fig, ax = plt.subplots(figsize=fig_size)
	        sns.scatterplot(data=v_svdvals.iloc[:200, :], s=15).set(title=f"{using_transformer} - Singular values of the V matrix in the {epoch}th epoch", ylabel="Singular Value", xlabel = "Position of Singular Value")
	        plt.savefig(f"{output}v.png", dpi=750)
	        plt.close()
        
def graphics_sv_layers_blocks(models_dict, using_transformer, fig_size, layers):
    for layer in range(4):
        if using_transformer == 'CSWin':
            layer += 1
        layer = str(layer)
        num_blocks = layers[using_transformer][layer][0]
        mat_size = layers[using_transformer][layer][1]
        if using_transformer == "CSWin":
            output = f"grafics/{using_transformer}_stage{layer}_singular-values_"
        elif using_transformer == "SWin":
            output = f"grafics/{using_transformer}_layer-{layer}_singular-values_"
            
        q_svdvals_means = pd.DataFrame()
        k_svdvals_means = pd.DataFrame()
        v_svdvals_means = pd.DataFrame()
        for epoch, model in enumerate(models_dict):
            q11 = []
            k11 = []
            v11 = []
            q_svdvals = pd.DataFrame()
            k_svdvals = pd.DataFrame()
            v_svdvals = pd.DataFrame()
            for n in range(num_blocks):
                if using_transformer == "CSWin":
                    i = f'stage{layer}.{n}.qkv.weight'
                elif using_transformer == "SWin":
                    i = f'layers.{layer}.blocks.{n}.attn.qkv.weight'
                for j in model:
                    if i==j:
                        q11.append(model[j][:mat_size])
                        k11.append(model[j][mat_size:mat_size*2])
                        v11.append(model[j][mat_size*2:mat_size*3])
                for l in range(len(q11)):
                    q_svdvals[f"block{l}"] = torch.linalg.svdvals(q11[l])
                    k_svdvals[f"block{l}"] = torch.linalg.svdvals(k11[l])
                    v_svdvals[f"block{l}"] = torch.linalg.svdvals(v11[l])
            if using_transformer == "CSWin":
                q_svdvals_means[f"e{epoch}"] = q_svdvals.mean(axis=1)
                k_svdvals_means[f"e{epoch}"] = k_svdvals.mean(axis=1)
                v_svdvals_means[f"e{epoch}"] = v_svdvals.mean(axis=1) 
            elif using_transformer == "SWin":
                q_svdvals_means[f"e{epoch}"] = q_svdvals.mean(axis=1)
                k_svdvals_means[f"e{epoch}"] = k_svdvals.mean(axis=1)
                v_svdvals_means[f"e{epoch}"] = v_svdvals.mean(axis=1)
        # plot the graphs
        q_svdvals = q_svdvals_means
        k_svdvals = k_svdvals_means
        v_svdvals = v_svdvals_means
        fig, ax = plt.subplots(figsize=fig_size)
        if q_svdvals.shape[1] > 12:
            indexs = [i for i in range(q_svdvals.shape[1]) if i%4==0]
        else:
            indexs = [i for i in range(q_svdvals.shape[1])]
        if using_transformer == 'CSWin': layer = 'stage'
        elif using_transformer == 'SWin': layer = 'layer'
        sns.scatterplot(data=q_svdvals.iloc[:200, indexs], s=15).set(title=f"{using_transformer} - Evolution of the singular values of the Q matrix in the {layer} {n}", ylabel="Singular Value", xlabel = "Position of Singular Value")
        plt.savefig(f"{output}q.png", dpi=750)
        plt.close()

        fig, ax = plt.subplots(figsize=fig_size)
        if k_svdvals.shape[1] > 12:
            indexs = [i for i in range(k_svdvals.shape[1]) if i%4==0]
        else:
            indexs = [i for i in range(k_svdvals.shape[1])]
        sns.scatterplot(data=k_svdvals.iloc[:200, indexs], s=15).set(title=f"{using_transformer} - Evolution of the singular values of the K matrix in the {layer} {n}", ylabel="Singular Value", xlabel = "Position of Singular Value")
        plt.savefig(f"{output}k.png", dpi=750)
        plt.close()

        fig, ax = plt.subplots(figsize=fig_size)
        if v_svdvals.shape[1] > 12:
            indexs = [i for i in range(v_svdvals.shape[1]) if i%4==0]
        else:
            indexs = [i for i in range(v_svdvals.shape[1])]
        sns.scatterplot(data=v_svdvals.iloc[:200, indexs], s=15).set(title=f"{using_transformer} - Evolution of the singular values of the V matrix in the {layer} {n}", ylabel="Singular Value", xlabel = "Position of Singular Value")
        plt.savefig(f"{output}v.png", dpi=750)
        plt.close()

def graphics_mlp_blocks(models_dict, using_transformer, mat_size, num_blocks, fig_size, layer=False):
    for mlp_block in [1,2]:
#        for epoch, model in enumerate(models_dict): # the epoch is fixed in each graphic
#        	if epoch%4 ==0:
#	            mlp = []
#	            mlp_svdvals = pd.DataFrame()
#	            if using_transformer == "CSWin":
#	                output = f"grafics/{using_transformer}_stage{layer}_epoch-{epoch}_singular-values_mlp_{mlp_block}"
#	            elif using_transformer == "SWin":
#	                output = f"grafics/{using_transformer}_layer-{layer}_epoch-{epoch}_singular-values_mlp_{mlp_block}"
#	            else:
#	                output = f"grafics/{using_transformer}_epoch-{epoch}_singular-values_mlp_{mlp_block}"
#	            for n in range(num_blocks):
#	                if using_transformer == "CSWin":
#	                    i = f'stage{layer}.{n}.mlp.fc{mlp_block}.weight'
#	                elif using_transformer == "SWin":
#	                    i = f'layers.{layer}.blocks.{n}.mlp.fc{mlp_block}.weight'
#	                else:
#	                    i = f'blocks.{n}.mlp.fc{mlp_block}.weight'
#	                for j in model:
#	                    if i==j:
#	                        mlp.append(model[j])
#	                for l in range(len(mlp)):
#	                    mlp_svdvals[f"block{l}"] = torch.linalg.svdvals(mlp[l])
#	            # make the heatmaps
#	            mlp_svdvals = mlp_svdvals
#	            fig, ax = plt.subplots(figsize=fig_size)
#	            sns.scatterplot(data=mlp_svdvals.iloc[:200, :], s=15).set(title=f"{using_transformer} - Singular values of the mlp {mlp_block} in the {epoch}th epoch", ylabel="Singular Value", xlabel = "Position of Singular Value")
#	            plt.savefig(f"{output}.png", dpi=750)
#	            plt.close()

#        mlp_ranks = {}
#        mlp_norms = {}
        mlp_diffs = {}
        for n in range(num_blocks):
            mlp = []
#            mlp_svdvals = pd.DataFrame()
#            mlp_ranks[f'block{n}'] = []
#            mlp_norms[f'block{n}'] = []
            mlp_diffs[f'block{n}'] = []
            if using_transformer == "CSWin":
                i = f'stage{layer}.{n}.mlp.fc{mlp_block}.weight'
                output = f"grafics/{using_transformer}_stage{layer}_mlp{mlp_block}"
            elif using_transformer == "SWin":
                i = f'layers.{layer}.blocks.{n}.mlp.fc{mlp_block}.weight'
                output = f"grafics/{using_transformer}_layer-{layer}_mlp{mlp_block}"
            else:
                i = f'blocks.{n}.mlp.fc{mlp_block}.weight'
                output = f"grafics/{using_transformer}_mlp{mlp_block}"
            for j in atributes_model[i]:
                mlp.append(j)
 #           for l in range(len(mlp)):
 #               mlp_svdvals[f"e{l}"] = torch.linalg.svdvals(mlp[l])
 #               mlp_ranks[f"block{n}"].append(torch.linalg.matrix_rank(mlp[l], tol=0.1).item())
 #               mlp_norms[f"block{n}"].append(torch.linalg.matrix_norm(mlp[l]).item())
            for l in range(1, len(mlp)):
                mlp_diffs[f"block{n}"].append(torch.sum(torch.abs(mlp[l]-mlp[l-1])).item())
                
            # plot the graphs
#            mlp_svdvals = mlp_svdvals
#            fig, ax = plt.subplots(figsize=fig_size)
#            if mlp_svdvals.shape[1] > 12:
#                indexs = [i for i in range(mlp_svdvals.shape[1]) if i%4==0]
#            else:
#                indexs = [i for i in range(mlp_svdvals.shape[1])]
#            sns.scatterplot(data=mlp_svdvals.iloc[:200, indexs], s=15).set(title=f"{using_transformer} - Singular values of the mlp {mlp_block} in the block {n}", ylabel="Singular Value", xlabel = "Position of Singular Value")
#            plt.savefig(f"{output}_bloc-{n}_singular_values.png", dpi=750)
#            plt.close()
        
        # plot the rank graphs
#        mlp_ranks = pd.DataFrame(mlp_ranks)
#        mlp_norms = pd.DataFrame(mlp_norms)
        mlp_diffs = pd.DataFrame(mlp_diffs)

#        fig, ax = plt.subplots(figsize=fig_size)
#        sns.lineplot(data=mlp_ranks.iloc[:200, :], markers=True, legend=False).set(title=f"{using_transformer} - Evolution of the ranks of the mlp {mlp_block} matrices", ylabel="Rank", xlabel = "Epoch")
#        plt.legend(labels=mlp_ranks.columns, loc='upper right')
#        plt.savefig(f"{output}_ranks.png", dpi=750)
#        plt.close()
        

#        fig, ax = plt.subplots(figsize=fig_size)
#        sns.lineplot(data=mlp_norms.iloc[:200, :], markers=True, legend=False).set(title=f"{using_transformer} - Evolution of the Frobenius norms of the mlp {mlp_block} matrices", ylabel="Norm", xlabel = "Epoch")
#        plt.legend(labels=mlp_norms.columns, loc='upper right')
#        plt.savefig(f"{output}_norms.png", dpi=750)
#        plt.close()
        
        fig, ax = plt.subplots(figsize=fig_size)
        sns.lineplot(data=mlp_diffs.iloc[:200, :], markers=True, legend=False).set(title=f"{using_transformer} - Evolution of the differences between consecutive epochs in the mlp {mlp_block} matrices", ylabel="Absolute difference", xlabel = "Epoch")
        plt.legend(labels=mlp_diffs.columns, loc='upper right')
        plt.savefig(f"{output}_diffs.png", dpi=750)
        plt.close()

def graphics_sv_layers_epochs_mlp(models_dict, using_transformer, fig_size, layers):
    for mlp_block in [1,2]:
        for epoch, model in enumerate(models_dict): # the epoch is fixed in each graphic
        	if epoch%4==0:
	            mlp_svdvals_means = pd.DataFrame()
	            for layer in range(4):
	                if using_transformer == 'CSWin':
	                    layer += 1
	                layer = str(layer)
	                mlp = []
	                mlp_svdvals = pd.DataFrame()
	                num_blocks = layers[using_transformer][layer][0]
	                mat_size = layers[using_transformer][layer][1]
	                if using_transformer == "CSWin":
	                    output = f"grafics/{using_transformer}_epoch-{epoch}_singular-values_mlp{mlp_block}"
	                elif using_transformer == "SWin":
	                    output = f"grafics/{using_transformer}_epoch-{epoch}_singular-values_mlp{mlp_block}"
	                for n in range(num_blocks):
	                    if using_transformer == "CSWin":
	                        i = f'stage{layer}.{n}.mlp.fc{mlp_block}.weight'
	                    elif using_transformer == "SWin":
	                        i = f'layers.{layer}.blocks.{n}.mlp.fc{mlp_block}.weight'
	                    for j in model:
	                        if i==j:
	                            mlp.append(model[j])
	                    for l in range(len(mlp)):
	                        mlp_svdvals[f"block{l}"] = torch.linalg.svdvals(mlp[l])
	                if using_transformer == "CSWin":
	                    mlp_svdvals_means[f"stage{layer}"] = mlp_svdvals.mean(axis=1)
	                elif using_transformer == "SWin":
	                    mlp_svdvals_means[f"layer{layer}"] = mlp_svdvals.mean(axis=1)

	            # make the heatmaps
	            mlp_svdvals = mlp_svdvals_means
	            fig, ax = plt.subplots(figsize=fig_size)
	            sns.scatterplot(data=mlp_svdvals.iloc[:200, :], s=15).set(title=f"{using_transformer} - Singular values of the MLP {mlp_block} in the {epoch}th epoch", ylabel="Singular Value", xlabel = "Position of Singular Value")
	            plt.savefig(f"{output}.png", dpi=750)
	            plt.close()
        
def graphics_sv_layers_blocks_mlp(models_dict, using_transformer, fig_size, layers):
    for mlp_block in [1,2]:
        for layer in range(4):
            if using_transformer == 'CSWin':
                layer += 1
            layer = str(layer)
            num_blocks = layers[using_transformer][layer][0]
            mat_size = layers[using_transformer][layer][1]
            if using_transformer == "CSWin":
                output = f"grafics/{using_transformer}_stage{layer}_singular-values_mlp{mlp_block}"
            elif using_transformer == "SWin":
                output = f"grafics/{using_transformer}_layer-{layer}_singular-values_mlp{mlp_block}"

            mlp_svdvals_means = pd.DataFrame()
            for epoch, model in enumerate(models_dict):
                mlp = []
                mlp_svdvals = pd.DataFrame()
                for n in range(num_blocks):
                    if using_transformer == "CSWin":
                        i = f'stage{layer}.{n}.mlp.fc{mlp_block}.weight'
                    elif using_transformer == "SWin":
                        i = f'layers.{layer}.blocks.{n}.mlp.fc{mlp_block}.weight'
                    for j in model:
                        if i==j:
                            mlp.append(model[j])
                    for l in range(len(mlp)):
                        mlp_svdvals[f"block{l}"] = torch.linalg.svdvals(mlp[l])
                if using_transformer == "CSWin":
                    mlp_svdvals_means[f"e{epoch}"] = mlp_svdvals.mean(axis=1)
                elif using_transformer == "SWin":
                    mlp_svdvals_means[f"e{epoch}"] = mlp_svdvals.mean(axis=1)
            # plot the graphs
            mlp_svdvals = mlp_svdvals_means
            fig, ax = plt.subplots(figsize=fig_size)
            if using_transformer == 'CSWin': layer = 'stage'
            elif using_transformer == 'SWin': layer = 'layer'
            if mlp_svdvals.shape[1] > 12:
                indexs = [i for i in range(mlp_svdvals.shape[1]) if i%4==0]
            else:
                indexs = [i for i in range(mlp_svdvals.shape[1])]
            sns.scatterplot(data=mlp_svdvals.iloc[:200, indexs], s=15).set(title=f"{using_transformer} - Evolution of the singular values of the MLP {mlp_block} in the {layer} {n}", ylabel="Singular Value", xlabel = "Position of Singular Value")
            plt.savefig(f"{output}.png", dpi=750)
            plt.close()

def graphics_w0_blocks(models_dict, using_transformer, mat_size, num_blocks, fig_size, layer=False):
#    for epoch, model in enumerate(models_dict): # the epoch is fixed in each graphic
#    	if epoch%4==0:
#	        w0 = []
#	        w0_svdvals = pd.DataFrame()
#	        if using_transformer == "CSWin":
#	            output = f"grafics/{using_transformer}_stage{layer}_epoch-{epoch}_singular-values_w0"
#	        elif using_transformer == "SWin":
#	            output = f"grafics/{using_transformer}_layer-{layer}_epoch-{epoch}_singular-values_w0"
#	        else:
#	            output = f"grafics/{using_transformer}_epoch-{epoch}_singular-values_w0"
#	        for n in range(num_blocks):
#	            if using_transformer == "CSWin":
#	                i = f'stage{layer}.{n}.proj.weight'
#	            elif using_transformer == "SWin":
#	                i = f'layers.{layer}.blocks.{n}.attn.proj.weight'
#	            else:
#	                i = f'blocks.{n}.attn.proj.weight'
#	            for j in model:
#	                if i==j:
#	                    w0.append(model[j])
#	            for l in range(len(w0)):
#	                w0_svdvals[f"block{l}"] = torch.linalg.svdvals(w0[l])
#	        # make the heatmaps
#	        w0_svdvals = w0_svdvals
#	        fig, ax = plt.subplots(figsize=fig_size)
#	        sns.scatterplot(data=w0_svdvals.iloc[:200, :], s=15).set(title=f"{using_transformer} - Singular values of the W0 matrices in the {epoch}th epoch", ylabel="Singular Value", xlabel = "Position of Singular Value")
#	        plt.savefig(f"{output}.png", dpi=750)
#	        plt.close()

#    w0_ranks = {}
#    w0_norms = {}
    w0_diffs = {}
    for n in range(num_blocks):
        w0 = []
        w0_svdvals = pd.DataFrame()
#        w0_ranks[f'block{n}'] = []
#        w0_norms[f'block{n}'] = []
        w0_diffs[f'block{n}'] = []
        if using_transformer == "CSWin":
            i = f'stage{layer}.{n}.proj.weight'
            output = f"grafics/{using_transformer}_stage{layer}_w0"
        elif using_transformer == "SWin":
            i = f'layers.{layer}.blocks.{n}.attn.proj.weight'
            output = f"grafics/{using_transformer}_layer-{layer}_w0"
        else:
            i = f'blocks.{n}.attn.proj.weight'
            output = f"grafics/{using_transformer}_w0"
        for j in atributes_model[i]:
            w0.append(j)
#        for l in range(len(w0)):
#            w0_svdvals[f"e{l}"] = torch.linalg.svdvals(w0[l])
#            w0_ranks[f"block{n}"].append(torch.linalg.matrix_rank(w0[l], tol=0.1).item())
#            w0_norms[f"block{n}"].append(torch.linalg.matrix_norm(w0[l]).item())
        for l in range(1, len(w0)):
            w0_diffs[f"block{n}"].append(torch.sum(torch.abs(w0[l]-w0[l-1])).item())

        # plot the graphs
#        w0_svdvals = w0_svdvals
#        fig, ax = plt.subplots(figsize=fig_size)
#        if w0_svdvals.shape[1] > 12:
#            indexs = [i for i in range(w0_svdvals.shape[1]) if i%4==0]
#        else:
#            indexs = [i for i in range(w0_svdvals.shape[1])]
#        sns.scatterplot(data=w0_svdvals.iloc[:200, indexs], s=15).set(title=f"{using_transformer} - Singular values of the W0 matrices in the block {n}", ylabel="Singular Value", xlabel = "Position of Singular Value")
#        plt.savefig(f"{output}_bloc-{n}_singular_values.png", dpi=750)
#        plt.close()

    # plot the rank graphs
#    w0_ranks = pd.DataFrame(w0_ranks)
#    w0_norms = pd.DataFrame(w0_norms)
    w0_diffs = pd.DataFrame(w0_diffs)

#    fig, ax = plt.subplots(figsize=fig_size)
#    sns.lineplot(data=w0_ranks.iloc[:200, :], markers=True, legend=False).set(title=f"{using_transformer} - Evolution of the ranks of the W0 matrices", ylabel="Rank", xlabel = "Epoch")
#    plt.legend(labels=w0_ranks.columns, loc='upper right')
#    plt.savefig(f"{output}_ranks.png", dpi=750)
#    plt.close()


#    fig, ax = plt.subplots(figsize=fig_size)
#    sns.lineplot(data=w0_norms.iloc[:200, :], markers=True, legend=False).set(title=f"{using_transformer} - Evolution of the Frobenius norms of the W0 matrices", ylabel="Norm", xlabel = "Epoch")
#    plt.legend(labels=w0_norms.columns, loc='upper right')
#    plt.savefig(f"{output}_norms.png", dpi=750)
    plt.close()

    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(data=w0_diffs.iloc[:200, :], markers=True, legend=False).set(title=f"{using_transformer} - Evolution of the differences between consecutive epochs in the W0 matrices", ylabel="Absolute difference", xlabel = "Epoch")
    plt.legend(labels=w0_diffs.columns, loc='upper right')
    plt.savefig(f"{output}_diffs.png", dpi=750)
    plt.close()

def graphics_sv_layers_epochs_w0(models_dict, using_transformer, fig_size, layers):
    for epoch, model in enumerate(models_dict): # the epoch is fixed in each graphic
    	if epoch%4==0:
	        w0_svdvals_means = pd.DataFrame()
	        for layer in range(4):
	            if using_transformer == 'CSWin':
	                layer += 1
	            layer = str(layer)
	            w0 = []
	            w0_svdvals = pd.DataFrame()
	            num_blocks = layers[using_transformer][layer][0]
	            mat_size = layers[using_transformer][layer][1]
	            if using_transformer == "CSWin":
	                output = f"grafics/{using_transformer}_epoch-{epoch}_singular-values_w0"
	            elif using_transformer == "SWin":
	                output = f"grafics/{using_transformer}_epoch-{epoch}_singular-values_w0"
	            for n in range(num_blocks):
	                if using_transformer == "CSWin":
	                    i = f'stage{layer}.{n}.proj.weight'
	                elif using_transformer == "SWin":
	                    i = f'layers.{layer}.blocks.{n}.attn.proj.weight'
	                for j in model:
	                    if i==j:
	                        w0.append(model[j])
	                for l in range(len(w0)):
	                    w0_svdvals[f"block{l}"] = torch.linalg.svdvals(w0[l])
	            if using_transformer == "CSWin":
	                w0_svdvals_means[f"stage{layer}"] = w0_svdvals.mean(axis=1)
	            elif using_transformer == "SWin":
	                w0_svdvals_means[f"layer{layer}"] = w0_svdvals.mean(axis=1)

	        # make the heatmaps
	        w0_svdvals = w0_svdvals_means
	        fig, ax = plt.subplots(figsize=fig_size)
	        sns.scatterplot(data=w0_svdvals.iloc[:200, :], s=15).set(title=f"{using_transformer} - Singular values of the W0 in the {epoch}th epoch", ylabel="Singular Value", xlabel = "Position of Singular Value")
	        plt.savefig(f"{output}.png", dpi=750)
	        plt.close()
        
def graphics_sv_layers_blocks_w0(models_dict, using_transformer, fig_size, layers):
    for layer in range(4):
        if using_transformer == 'CSWin':
            layer += 1
        layer = str(layer)
        num_blocks = layers[using_transformer][layer][0]
        mat_size = layers[using_transformer][layer][1]
        if using_transformer == "CSWin":
            output = f"grafics/{using_transformer}_stage{layer}_singular-values_w0"
        elif using_transformer == "SWin":
            output = f"grafics/{using_transformer}_layer-{layer}_singular-values_w0"

        w0_svdvals_means = pd.DataFrame()
        for epoch, model in enumerate(models_dict):
            w0 = []
            w0_svdvals = pd.DataFrame()
            for n in range(num_blocks):
                if using_transformer == "CSWin":
                    i = f'stage{layer}.{n}.proj.weight'
                elif using_transformer == "SWin":
                    i = f'layers.{layer}.blocks.{n}.attn.proj.weight'
                for j in model:
                    if i==j:
                        w0.append(model[j])
                for l in range(len(w0)):
                    w0_svdvals[f"block{l}"] = torch.linalg.svdvals(w0[l])
            if using_transformer == "CSWin":
                w0_svdvals_means[f"e{epoch}"] = w0_svdvals.mean(axis=1)
            elif using_transformer == "SWin":
                w0_svdvals_means[f"e{epoch}"] = w0_svdvals.mean(axis=1)
        # plot the graphs
        w0_svdvals = w0_svdvals_means
        fig, ax = plt.subplots(figsize=fig_size)
        if w0_svdvals.shape[1] > 12:
            indexs = [i for i in range(w0_svdvals.shape[1]) if i%4==0]
        else:
            indexs = [i for i in range(w0_svdvals.shape[1])]
        sns.scatterplot(data=w0_svdvals.iloc[:200, indexs], s=15).set(title=f"{using_transformer} - Evolution of the singular values of the W0 matrices in the {layer} {n}", ylabel="Singular Value", xlabel = "Position of Singular Value")
        plt.savefig(f"{output}.png", dpi=750)
        plt.close()

def graphics_global_variables(models_dict, using_transformer, mat_size, num_blocks, fig_size, has_positional_embedding, has_distillation_token,):
    ct = []
    if has_positional_embedding: pe = []
    if has_distillation_token: dt = []
    for model in models_dict:
        ct.append(model['cls_token'])
        if has_positional_embedding: pe.append(model['pos_embed'])
        if has_distillation_token: dt.append(model['dist_token'])
    output = f"grafics/{using_transformer}"
    
    # make graphs differences
    ct_diffs = {}
    if has_positional_embedding: pe_diffs = {}
    if has_distillation_token: dt_diffs = {}
    for epoch in range(1, len(ct)):
        ct_diffs[f'{epoch}'] = [torch.sum(torch.abs(ct[epoch]-ct[epoch-1])).item()]
        if has_positional_embedding: pe_diffs[f'{epoch}'] = [torch.sum(torch.abs(pe[epoch]-pe[epoch-1])).item()]
        if has_distillation_token: dt_diffs[f'{epoch}'] = [torch.sum(torch.abs(dt[epoch]-dt[epoch-1])).item()]
    
    ct_diffs = pd.DataFrame(ct_diffs).T
    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(data=ct_diffs, markers=True, legend=False).set(title=f"{using_transformer} - Evolution of the differences between consecutive epochs in the class token matrix", ylabel="Absolute difference", xlabel = "Epoch")
    #plt.legend(labels=ct_diffs.columns, loc='upper right')
    plt.savefig(f"{output}_ct_diffs.png", dpi=750)
    plt.close()
    
    if has_positional_embedding: 
        pe_diffs = pd.DataFrame(pe_diffs).T
        fig, ax = plt.subplots(figsize=fig_size)
        sns.lineplot(data=pe_diffs, markers=True, legend=False).set(title=f"{using_transformer} - Evolution of the differences between consecutive epochs in the positional embedding matrix", ylabel="Absolute difference", xlabel = "Epoch")
        #plt.legend(labels=ct_diffs.columns, loc='upper right')
        plt.savefig(f"{output}_pe_diffs.png", dpi=750)
        plt.close()
    
    if has_distillation_token:
        dt_diffs = pd.DataFrame(dt_diffs).T
        fig, ax = plt.subplots(figsize=fig_size)
        sns.lineplot(data=dt_diffs, markers=True, legend=False).set(title=f"{using_transformer} - Evolution of the differences between consecutive epochs in the distillation token matrix", ylabel="Absolute difference", xlabel = "Epoch")
        #plt.legend(labels=ct_diffs.columns, loc='upper right')
        plt.savefig(f"{output}_dt_diffs.png", dpi=750)
        plt.close()

def graphics_determinants(models_dict, using_transformer, mat_size, num_blocks, fig_size, layer=False):
    q_dets = {}
    k_dets = {}
    v_dets = {}
    w0_dets = {}
    mlp1_dets = {}
    mlp2_dets = {}
    for n in range(num_blocks):
        q11 = []
        k11 = []
        v11 = []
        w0_11 = []
        mlp1_11 = []
        mlp2_11 = []
        q_dets[f'block{n}'] = []
        k_dets[f'block{n}'] = []
        v_dets[f'block{n}'] = []
        w0_dets[f'block{n}'] = []
        mlp1_dets[f'block{n}'] = []
        mlp2_dets[f'block{n}'] = []
        if using_transformer == "CSWin":
            i_qkv = f'stage{layer}.{n}.qkv.weight'
            i_w0 = f'stage{layer}.{n}.proj.weight'
            i_mlp1 = f'stage{layer}.{n}.mlp.fc1.weight'
            i_mlp2 = f'stage{layer}.{n}.mlp.fc2.weight'
            op_q = f"grafics/{using_transformer}_stage{layer}_det_q"
            op_k = f"grafics/{using_transformer}_stage{layer}_det_k"
            op_v = f"grafics/{using_transformer}_stage{layer}_det_v"
            op_mlp1 = f"grafics/{using_transformer}_stage{layer}_det_mlp1"
            op_mlp2 = f"grafics/{using_transformer}_stage{layer}_det_mlp2"
            op_w0 = f"grafics/{using_transformer}_stage{layer}_det_w0"
        elif using_transformer == "SWin":
            i_qkv = f'layers.{layer}.blocks.{n}.attn.qkv.weight'
            i_w0 = f'layers.{layer}.blocks.{n}.attn.proj.weight'
            i_mlp1 = f'layers.{layer}.blocks.{n}.mlp.fc1.weight'
            i_mlp2 = f'layers.{layer}.blocks.{n}.mlp.fc2.weight'
            op_q = f"grafics/{using_transformer}_layer-{layer}_det_q"
            op_k = f"grafics/{using_transformer}_layer-{layer}_det_k"
            op_v = f"grafics/{using_transformer}_layer-{layer}_det_v"
            op_w0 = f"grafics/{using_transformer}_layer-{layer}_det_w0"
            op_mlp1 = f"grafics/{using_transformer}_layer-{layer}_det_mlp1"
            op_mlp2 = f"grafics/{using_transformer}_layer-{layer}_det_mlp2"
        else:
            i_qkv = f'blocks.{n}.attn.qkv.weight'
            i_w0 = f'blocks.{n}.attn.proj.weight'
            i_mlp1 = f'blocks.{n}.mlp.fc1.weight'
            i_mlp2 = f'blocks.{n}.mlp.fc2.weight'
            op_q = f"grafics/{using_transformer}_det_q"
            op_k = f"grafics/{using_transformer}_det_k"
            op_v = f"grafics/{using_transformer}_det_v"
            op_w0 = f"grafics/{using_transformer}_det_w0"
            op_mlp1 = f"grafics/{using_transformer}_det_mlp1"
            op_mlp2 = f"grafics/{using_transformer}_det_mlp2"
        for j in atributes_model[i_qkv]:
            q = j[:mat_size]
            k = j[mat_size:mat_size*2]
            v = j[mat_size*2:mat_size*3]
            q11.append(q)
            k11.append(k)
            v11.append(v)
        for j in atributes_model[i_w0]:
            w0_11.append(j)
        for j in atributes_model[i_mlp1]:
            mlp1_11.append(j)
        for j in atributes_model[i_mlp2]:
            mlp2_11.append(j)
        for l in range(len(q11)):
            q_dets[f"block{n}"].append(torch.linalg.det(q11[l]).item())
            k_dets[f"block{n}"].append(torch.linalg.det(k11[l]).item())
            v_dets[f"block{n}"].append(torch.linalg.det(v11[l]).item())
            w0_dets[f"block{n}"].append(torch.linalg.det(w0_11[l]).item())

    # plot the graphs
    q_dets = pd.DataFrame(q_dets)
    k_dets = pd.DataFrame(k_dets)
    v_dets = pd.DataFrame(v_dets)
    w0_dets = pd.DataFrame(w0_dets)

    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(data=q_dets.iloc[:200, :], markers=True, legend=False).set(title=f"{using_transformer} - Evolution of the determinants of the W_Q matrices", ylabel="Determinant", xlabel = "Epoch")
    plt.legend(labels=q_dets.columns, loc='upper right')
    plt.savefig(f"{op_q}.png", dpi=750)
    plt.close()

    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(data=k_dets.iloc[:200, :], markers=True, legend=False).set(title=f"{using_transformer} - Evolution of the determinants of the W_K matrices", ylabel="Determinant", xlabel = "Epoch")
    plt.legend(labels=k_dets.columns, loc='upper right')
    plt.savefig(f"{op_k}.png", dpi=750)
    plt.close()

    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(data=v_dets.iloc[:200, :], markers=True, legend=False).set(title=f"{using_transformer} - Evolution of the determinants of the W_V matrices", ylabel="Determinant", xlabel = "Epoch")
    plt.legend(labels=v_dets.columns, loc='upper right')
    plt.savefig(f"{op_v}.png", dpi=750)
    plt.close()

    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(data=w0_dets.iloc[:200, :], markers=True, legend=False).set(title=f"{using_transformer} - Evolution of the determinants of the W0 matrices", ylabel="Determinant", xlabel = "Epoch")
    plt.legend(labels=w0_dets.columns, loc='upper right')
    plt.savefig(f"{op_w0}.png", dpi=750)
    plt.close()

if using_transformer in ['ViT', 'BEiT', 'DeiT']:
    mat_size = 768
    num_blocks = 12
    # print('starting sv blocks')
    # graphics_sv_blocks(models_dict, using_transformer, mat_size, num_blocks, fig_size)
    # print('starting sv epochs')
    # graphics_sv_epochs(models_dict, using_transformer, mat_size, num_blocks, fig_size)
    # print('starting ranks')
    # graphics_ranks(models_dict, using_transformer, mat_size, num_blocks, fig_size)
    # print('starting norms')
    # graphics_norms(models_dict, using_transformer, mat_size, num_blocks, fig_size)
    print('starting diff epochs')
    graphics_diff_epochs(models_dict, using_transformer, mat_size, num_blocks, fig_size)
    # print('starting mlp blocks')
    # graphics_mlp_blocks(models_dict, using_transformer, mat_size, num_blocks, fig_size)
    # print('starting w0 blocks')
    # graphics_w0_blocks(models_dict, using_transformer, mat_size, num_blocks, fig_size)
    # if using_transformer == 'ViT':
    #     pe, dt = True, False
    # elif using_transformer == 'BEiT':
    #     pe, dt = False, False
    # elif using_transformer == 'DeiT':
    #     pe, dt = True, True
    # print('starting global variables')
    # graphics_global_variables(models_dict, using_transformer, mat_size, num_blocks, fig_size, pe, dt)
    # print('starting graphics deteminants')
    # graphics_determinants(models_dict, using_transformer, mat_size, num_blocks, fig_size)

if using_transformer in ['CSWin', 'SWin']:
    for layer in range(4):
        if using_transformer == 'CSWin':
            layer += 1
        layer = str(layer)
        num_blocks = layers[using_transformer][layer][0]
        mat_size = layers[using_transformer][layer][1]
        # print('starting graphics determinants')
        # graphics_determinants(models_dict, using_transformer, mat_size, num_blocks, fig_size, layer)
        # print('starting sv blocks')
        # graphics_sv_blocks(models_dict, using_transformer, mat_size, num_blocks, fig_size, layer)
        # print('starting sv epochs')
        # graphics_sv_epochs(models_dict, using_transformer, mat_size, num_blocks, fig_size, layer)
        # print('starting ranks qkv')
        # graphics_ranks(models_dict, using_transformer, mat_size, num_blocks, fig_size, layer)
        # print('starting norms qkv')
        # graphics_norms(models_dict, using_transformer, mat_size, num_blocks, fig_size, layer)
        print('starting diffs qkv')
        graphics_diff_epochs(models_dict, using_transformer, mat_size, num_blocks, fig_size, layer)
        # print('starting mlp')
        # graphics_mlp_blocks(models_dict, using_transformer, mat_size, num_blocks, fig_size, layer)
        # print('starting w0')
        # graphics_w0_blocks(models_dict, using_transformer, mat_size, num_blocks, fig_size, layer)
    # print('starting sv layers blocks qkv')
    # graphics_sv_layers_blocks(models_dict, using_transformer, fig_size, layers)
    # print('starting sv layers epochs qkv')
    # graphics_sv_layers_epochs(models_dict, using_transformer, fig_size, layers)
    # print('starting sv layers blocks mlp')
    # graphics_sv_layers_blocks_mlp(models_dict, using_transformer, fig_size, layers)
    # print('starting sv layers epochs mlp')
    # graphics_sv_layers_epochs_mlp(models_dict, using_transformer, fig_size, layers)
    # print('starting sv layers blocks w0')
    # graphics_sv_layers_blocks_w0(models_dict, using_transformer, fig_size, layers)
    # print('starting sv layers epochs w0')
    # graphics_sv_layers_epochs_w0(models_dict, using_transformer, fig_size, layers)
