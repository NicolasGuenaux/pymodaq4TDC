import os
import numpy as np
os.chdir(r"C:\Users\NG278231\Documents\These\TDC")
data = np.loadtxt("tmp_N=1_t=100_100kHz.txt")
print(data[:,3]*27.4e-6)

import h5py


def save_to_h5_multiple_subsets(filename, new_data, subset_name=None):
    """
    Save new_data as a separate subset in the H5 file.

    Parameters:
        filename (str): Path to the H5 file.
        new_data (numpy.ndarray): Data to save.
        subset_name (str): Name of the subset in the H5 file. If None, an incremental name will be used.
    """
    with h5py.File(filename, "a") as h5_file:  # "a" mode for appending or creating
        # Generate a unique subset name if not provided
        if subset_name is None:
            subset_name = f"subset_{len(h5_file.keys()) + 1}"

        if subset_name in h5_file:
            raise ValueError(f"Subset name '{subset_name}' already exists in the file.")

        # Save the new data under the unique subset name
        h5_file.create_dataset(subset_name, data=new_data)
        print(f"Data saved under subset '{subset_name}' in {filename}")


# Example usage
filename = "data_multiple_subsets.h5"

# Save new subsets
new_data1 = np.random.rand(10, 5)  # 10x5 array
save_to_h5_multiple_subsets(filename, new_data1)

new_data2 = np.random.rand(15, 15)  # Another 15x5 array
save_to_h5_multiple_subsets(filename, new_data2)

# Reading back all subsets
with h5py.File(filename, "r") as h5_file:
    print("Datasets in the file:")
    for subset_name in h5_file.keys():
        print(f" - {subset_name}: shape {h5_file[subset_name].shape}")

