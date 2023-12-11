from dataloader.loader import CustomDataset
from torch.utils.data import DataLoader
from model.mapping import EncoderDecoder
from executor.train import train
from executor.train import load_model_epoch
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from ops.signed_distance import curve_point_nurbs
from configreader.configreader import parse_config
import sys
from evaluations.evaluator import calculate_lpips,calculate_psnr,calculate_ssim
import imageio
# Assuming you have a function to convert torch tensor to NumPy array
def to_numpy(tensor):
    return tensor.cpu().detach().numpy() if tensor.requires_grad else tensor.cpu().numpy()
# the file is in experiments/splinetofield/executor/train.py

def train_func(batch_size,num_epochs,train_data_cp,train_data_sdf,test_data_cp,test_data_sdf,save_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EncoderDecoder(train_data_cp[0].shape, hidden_layer_size=[16,14,12], latent_dim=16,output_size=128).to(device)
    custom_dataset = CustomDataset(train_data_cp.to(device), train_data_sdf.to(device),True)
    train_loader = DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=True)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    train(train_loader, model, optimizer, criterion, device,num_epochs,save_path,test_data_cp,test_data_sdf)

def test_func(test_data_cp,test_data_sdf,save_path):
    print("Testing the model")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EncoderDecoder(test_data_cp[0].shape, hidden_layer_size=[16,14,12], latent_dim=16,output_size=128).to(device)
    print(test_data_cp.shape)
    # load model
    model = load_model_epoch(model,save_path)
    model.eval()
    model.to(device)
    test_data_cp.to(device)
    test_data_sdf.to(device)
    # # evaluate on test data
    print("Evaluating on test data")
    computed_sdf = model(test_data_cp)
    loss = torch.nn.MSELoss()
    loss_val = loss(computed_sdf,test_data_sdf)
    print(f"Loss on test data is {loss_val}")
    # calculate lpips
    lpips = calculate_lpips(computed_sdf,test_data_sdf)
    # compute the mean lpips
    print(f"Mean lpips is {lpips}")
    # calculate ssim

    # calculate psnr
    np_computed_sdf = to_numpy(computed_sdf)
    np_test_data_sdf = to_numpy(test_data_sdf)
    psnr = calculate_psnr(np_computed_sdf,np_test_data_sdf)
    print(f"Mean psnr is {psnr}")
    # Assuming you have a function to convert torch tensor to NumPy array
    ssim = calculate_ssim(np_computed_sdf,np_test_data_sdf)
    # compute the mean ssim
    print(f"Mean ssim is {ssim}")
    # Select 10 test samples
    selected_indices = [0, 1, 2, 3]
    selected_test_data_cp = test_data_cp[selected_indices]
    selected_test_data_sdf = test_data_sdf[selected_indices]

    # Evaluate on selected test data
    computed_sdf = model(selected_test_data_cp.to(device))
    computed_sdf_np = to_numpy(computed_sdf)

    # Create a figure with subplots for each sample in the batch


    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # for i in range(len(selected_indices)):
    #     # Clear the previous plots
    #     for ax in axs:
    #         ax.clear()

    #     # Extract control points and reshape signed distance function
    #     x_data_i = to_numpy(selected_test_data_cp[i])
    #     y_data_i_true = to_numpy(selected_test_data_sdf[i])
    #     y_data_i_pred = computed_sdf_np[i]

    #     # Generate polygon
    #     polygon = curve_point_nurbs(x_data_i)

    #     # Plot the polygon
    #     axs[0].add_patch(Polygon(polygon, edgecolor='red', fill=None, linewidth=2))
    #     axs[0].set_xlim(-1, 1)
    #     axs[0].set_ylim(-1, 1)
    #     axs[0].set_title(f'Nurbs - Sample {selected_indices[i]}')

    #     # Plot the true signed distance field
    #     img_true = axs[1].imshow(y_data_i_true,origin='lower')
    #     axs[1].set_title(f'True Signed Distance - Sample {selected_indices[i]}')

    #     # Plot the predicted signed distance field
    #     img_pred = axs[2].imshow(y_data_i_pred[0], origin='lower')
    #     axs[2].set_title(f'Predicted Signed Distance - Sample {selected_indices[i]}')

    #     # Save the plot as an image file
    #     plt.tight_layout()
    #     filename = f'test_results_visualization_{i}.png'
    #     plt.savefig(filename)

    # # Create a list of filenames
    # # Create a list of filenames
    # filenames = [f'test_results_visualization_{i}.png' for i in range(len(selected_indices))]

    # # Convert the images into a GIF
    # images = [imageio.v2.imread(filename) for filename in filenames]
    # imageio.mimsave('test_results_visualization.gif', images, fps=1)  # Adjust fps (frames per second) as needed
    fig, axs = plt.subplots(len(selected_indices), 3, figsize=(15, 5 * len(selected_indices)))
    
    for i in range(len(selected_indices)):
        # Extract control points and reshape signed distance function
        x_data_i = to_numpy(selected_test_data_cp[i])
        y_data_i_true = to_numpy(selected_test_data_sdf[i])
        y_data_i_pred = computed_sdf_np[i]

        # Generate polygon
        polygon = curve_point_nurbs(x_data_i)

        # Plot the polygon
        axs[i, 0].add_patch(Polygon(polygon, edgecolor='red', fill=None, linewidth=2))
        axs[i, 0].set_xlim(-1, 1)
        axs[i, 0].set_ylim(-1, 1)
        axs[i, 0].set_title(f'Nurbs - Sample {selected_indices[i]}')

        # Plot the true signed distance field
        img_true = axs[i, 1].imshow(y_data_i_true,origin='lower')
        axs[i, 1].set_title(f'True Signed Distance - Sample {selected_indices[i]}')

        # Plot the predicted signed distance field
        img_pred = axs[i, 2].imshow(y_data_i_pred[0], origin='lower')
        axs[i, 2].set_title(f'Predicted Signed Distance - Sample {selected_indices[i]}')

    # # Add colorbars
    # cbar_true = fig.colorbar(img_true, ax=axs[:, 1], orientation='vertical', shrink=0.8, pad=0.02)
    # cbar_pred = fig.colorbar(img_pred, ax=axs[:, 2], orientation='vertical', shrink=0.8, pad=0.02)

    # Show the plot
    plt.tight_layout()
    plt.savefig('full_visualization.png')
    plt.show()
    # plot the results


if __name__ == '__main__':
    config_file_path = sys.argv[1]
    print(config_file_path)
    mode, batch_size, learning_rate, epochs, root_directory, save_path = parse_config(config_file_path)
    train_data_cp, test_data_cp, train_data_sdf, test_data_sdf = CustomDataset.return_test_train(root_directory)
    if mode == 'train':
        train_func(batch_size,epochs,train_data_cp,train_data_sdf,test_data_cp,test_data_sdf,save_path)
    elif mode == 'test':
        test_func(test_data_cp,test_data_sdf,save_path)
    else:
        print("Mode not supported")

