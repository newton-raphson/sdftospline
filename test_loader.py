from dataloader.loader import CustomDataset
from ops.signed_distance import curve_point_nurbs
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from torch.utils.data import DataLoader
import numpy as np
# # Example usage:
root_dir = '/work/mech-ai-scratch/samundra/experiments/splinetofield/experiment'
train_data_cp, test_data_cp, train_data_sdf, test_data_sdf = CustomDataset.return_test_train(root_dir)

# Create a CustomDataset instance
print(train_data_cp.shape)
custom_dataset = CustomDataset(train_data_cp, train_data_sdf, perturb_control_points=True)
# number of data points here 
print(len(custom_dataset))
# exit()
# Create DataLoader for training set
batch_size = 5
train_loader = DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=True)
# print(len(train_loader))
# exit()
batch_iterator = iter(train_loader)
batch = next(batch_iterator)
# Extract input and output data from the batch
x_data_np = batch['control_point'].numpy()
y_data_np = batch['signed_distance_function'].numpy()

# Create a figure with subplots for each sample in the batch
fig, axs = plt.subplots(2,len(x_data_np), figsize=(10, 5 * len(x_data_np)))

for i in range(len(x_data_np)):
    # Extract control points and reshape signed distance function
    x_data_i = x_data_np[i]
    print(x_data_i.shape)
    y_data_i = y_data_np[i]
    # Generate grid points
    x = np.linspace(-1, 1, 128)
    y = np.linspace(-1, 1, 128)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack((xx.flatten(), yy.flatten()))

    # Generate polygon
    polygon = curve_point_nurbs(x_data_i)

    # Plot the polygon
    axs[0, i].add_patch(Polygon(polygon, edgecolor='red', fill=None, linewidth=2))
    axs[0, i].set_xlim(-1, 1)
    axs[0, i].set_ylim(-1, 1)
    axs[0, i].set_title(f'Nurbs - Sample {i+1}')

    # Plot the image
    img_true = axs[1, i].imshow(y_data_i,origin='lower')
    # img = axs[i, 1].imshow(y_data_i, cmap='viridis', extent=(-1, 1, -1, 1))
    axs[1, i].set_title(f'Signed Distance - Sample {i+1}')

    # Highlight points with signed distances between -0.003 and 0.003
    chk = y_data_i.flatten()
    x = (chk > -0.003) & (chk < 0.003)
    highlighted_points = grid_points[x]
    axs[1, i].scatter(highlighted_points[:, 0], highlighted_points[:, 1], c='red', marker='.', label='Points in [-0.003, 0.003]')

    # # Add colorbar
    # cbar = fig.colorbar(img, ax=axs[i, 1], orientation='vertical')

# Show the plot
plt.tight_layout()
plt.savefig('batch_visualization.png')
