from dataloader.loader import CustomDataset
from torch.utils.data import DataLoader
import numpy as np
from model.mapping import EncoderDecoder
import os
import torch
import pickle
import os
import matplotlib.pyplot as plt
import glob
import re
from collections import OrderedDict

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def save_model(model, optimizer, loss_per_epoch, epoch,best_val_loss,val_loss_per_epoch, save_path):
    checkpoint_data = {
        'epoch': epoch,
        'loss_per_epoch': loss_per_epoch,
        'best_val_loss': best_val_loss,
        'val_loss_per_epoch': val_loss_per_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    checkpoint_path = os.path.join(save_path, f"train_model_epoch.pkl")
    with open(checkpoint_path, 'wb') as checkpoint_file:
        pickle.dump(checkpoint_data, checkpoint_file)
def save_model_epoch(model, epoch,save_path):
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    path=create_directory(os.path.join(save_path, "models"))
    checkpoint_path = os.path.join(path, f"model_epoch{epoch}.pkl")
    with open(checkpoint_path, 'wb') as checkpoint_file:
        pickle.dump(checkpoint_data, checkpoint_file)
def load_model(model, optimizer, save_path,):
    checkpoint_path = os.path.join(save_path, "train_model_epoch.pkl")
    with open(checkpoint_path, 'rb') as checkpoint_file:
        checkpoint_data = pickle.load(checkpoint_file)
    model.load_state_dict(checkpoint_data['model_state_dict'])
    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
    epoch = checkpoint_data['epoch']+1
    best_val_loss = checkpoint_data['best_val_loss']
    val_loss_per_epoch = checkpoint_data['val_loss_per_epoch']
    loss_per_epoch = checkpoint_data['loss_per_epoch']
    return model, optimizer, epoch, loss_per_epoch, best_val_loss, val_loss_per_epoch
import glob
import re

def load_model_epoch(model, save_path):
    # List all model files and find the one with the highest epoch number
    model_files = glob.glob(os.path.join(save_path, "models/model_epoch*.pkl"))
    latest_model_file = max(model_files, key=lambda x: int(re.findall(r'\d+', x)[-1]))

    with open(latest_model_file, 'rb') as checkpoint_file:
        checkpoint_data = pickle.load(checkpoint_file)

    # Create a new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in checkpoint_data['model_state_dict'].items():
        name = k[7:] if k.startswith('module.') else k  # remove `module.`
        new_state_dict[name] = v

    # Load the weights into the model
    model.load_state_dict(new_state_dict)
    return model

def train(train_loader, model, optimizer, criterion, device,num_epochs,save_path,test_cp,test_sdf):
    model.train()
    if os.path.exists(os.path.join(save_path,"train_model_epoch.pkl")):
        print("Resuming training from previous checkpoint")
        if device == 'cuda':
            torch.cuda.empty_cache()
        model, optimizer, start_epoch, loss_per_epoch,best_val_loss, val_loss_per_epoch = load_model(model, optimizer, save_path)
        print(f"Starting from epoch {start_epoch}")
        
    else:
        start_epoch = 0
        loss_per_epoch = []
        val_loss_per_epoch = []
        best_val_loss = float('inf')

    test_cp.to(device)
    test_sdf.to(device)
    for i in range(start_epoch, int(num_epochs)):
        train_loss = 0
        torch.cuda.empty_cache()
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            control_points = batch['control_point']
            signed_distance_functions = batch['signed_distance_function']
            output = model(control_points)
            output = torch.squeeze(output,dim=1)
            loss = criterion(output, signed_distance_functions)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader.dataset)
        loss_per_epoch.append(train_loss)
        # validation loss
        val_loss = criterion(model(test_cp),test_sdf)
        val_loss_per_epoch.append(val_loss.detach().numpy())
        print(f"Epoch {i+1} train loss: {train_loss:.10f} val loss: {val_loss:.10f}")
        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val_loss and i > 2:
            best_val_loss = val_loss
            save_model(model, optimizer, loss_per_epoch, i,best_val_loss,val_loss_per_epoch, save_path)
            torch.cuda.empty_cache()
            
        # regardless of the best loss, save the model every 50 epochs
        if (i+1) % 100 == 0:
            save_model_epoch(model, i+1, save_path)
            torch.cuda.empty_cache()
        if i%100==0:
            # plot val_loss and train_loss
            fig, ax = plt.subplots()
            ax.plot(loss_per_epoch, label='train_loss')
            ax.plot(val_loss_per_epoch, label='val_loss')
            ax.set_title('Loss vs Epochs')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            losspath=create_directory(os.path.join(save_path, "loss"))
            fig.savefig(os.path.join(losspath, f"loss{i}.png"))
        
    return train_loss
