import os
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    @staticmethod
    def return_test_train(root_dir, test_size=0.1, random_state=22):
        image_data = []  # Use a local variable to collect images
        for file in sorted(os.listdir(os.path.join(root_dir, "images"))):
            if file.endswith(".npy"):
                image = np.load(os.path.join(root_dir, "images", file), allow_pickle=True)
                image_data.append(image)

        # this loads all the control points
        data = np.array(image_data)
        control_points = np.load(os.path.join(root_dir, "lhs_ctrlpts.npy"), allow_pickle=True)
        control_points=control_points[:len(data)]
        train_data_cp, test_data_cp, train_data_sdf, test_data_sdf = train_test_split(
            control_points, data, test_size=test_size, random_state=random_state
        )
        # Convert NumPy arrays to PyTorch tensors
        return (
            torch.tensor(train_data_cp, dtype=torch.float32),
            torch.tensor(test_data_cp, dtype=torch.float32),
            torch.tensor(train_data_sdf, dtype=torch.float32),
            torch.tensor(test_data_sdf, dtype=torch.float32),
        )

    def __init__(self, control_points, signed_distance_functions):
        self.control_points = control_points
        self.signed_distance_functions = signed_distance_functions

    def __len__(self):
        return len(self.control_points)

    def __getitem__(self, idx):
        control_point = self.control_points[idx]
        sdf = self.signed_distance_functions[idx]
        return {"control_point": control_point, "signed_distance_function": sdf}

# Example usage:
root_dir = '/Users/samundrakarki/Desktop/ME625/FYP/splinetofield/dataloader/'
train_data_cp, test_data_cp, train_data_sdf, test_data_sdf = CustomDataset.return_test_train(root_dir)


# Create a CustomDataset instance
custom_dataset = CustomDataset(train_data_cp, train_data_sdf)

# Create DataLoader for training set
batch_size = 2
train_loader = DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=True)

# Assuming you have a DataLoader named train_loader
batch_iterator = iter(train_loader)
batch = next(batch_iterator)

#  conver the input and output to numpy
x_data_np = batch['control_point'].numpy()
y_data_np = batch['signed_distance_function'].numpy()
print(x_data_np.shape)
print(y_data_np.shape)

# # Plotting the input
# plt.subplot(1, 2, 1)
# plt.imshow(x_data_np[0, 0, :, :], cmap='gray')  # Assuming a grayscale image
# plt.title('Input Image')

# # Plotting the output
# plt.subplot(1, 2, 2)
# plt.imshow(y_data_np[0, 0, :, :], cmap='gray')  # Assuming a grayscale image
# plt.title('Output Image')

# plt.show()

