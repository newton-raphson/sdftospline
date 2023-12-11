import os
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    @staticmethod
    def return_test_train(root_dir, test_size=0.1, random_state=22):
        image_data = []  # Use a local variable to collect images
        files = os.listdir(os.path.join(root_dir, "images2"))
        for i in range(len(files)):
            image = np.load(os.path.join(root_dir, "images2", f"signed_distances{i}.npy"), allow_pickle=True)
            image_data.append(image)

            if len(image_data) == 1000:
                print("Loaded 19000 images")
                break
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

    def __init__(self, control_points, signed_distance_functions, perturb_control_points):
        if perturb_control_points:
            self.control_points, self.signed_distance_functions = self.increase_data(
                control_points, signed_distance_functions
            )
        else:
            self.control_points = control_points
            self.signed_distance_functions = signed_distance_functions
        # self.control_points = control_points
        # self.signed_distance_functions = signed_distance_functions
        # self.perturb_control_points = perturb_control_points

    def __len__(self):
        return len(self.control_points)

    def __getitem__(self, idx):
        control_point = self.control_points[idx]
        sdf = self.signed_distance_functions[idx]
        return {"control_point": control_point, "signed_distance_function": sdf}
        # Apply perturbation to control points


    def increase_data(self, control_points, sdf, perturbation_factor=0.001, num_times=20):
        augmented_control_points = control_points
        augmented_sdf = sdf

        for _ in range(num_times):
            # Generate random noise with the same shape as control_points
            noise = torch.rand_like(control_points) * 2 * perturbation_factor - perturbation_factor

            # Add the noise to control_points
            perturbed_control_points = control_points + noise
            perturbed_sdf = sdf

            # Concatenate perturbed data with the original data
            augmented_control_points = torch.cat((augmented_control_points, perturbed_control_points), dim=0)
            augmented_sdf = torch.cat((augmented_sdf, perturbed_sdf), dim=0)

        return augmented_control_points, augmented_sdf




