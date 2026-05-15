import kagglehub
import os

# Use dataset_download to download the entire dataset to your local cache
dataset_path = kagglehub.dataset_download("gopalbhattrai/pascal-voc-2012-dataset")

print(f"Dataset downloaded to: {dataset_path}")

# You can now explore the dataset with standard file/directory operations
# For example, list the directories in the dataset
dir_list = os.listdir(dataset_path)
print("Contents of the dataset:", dir_list)

# Let’s print the full path to the train_val directory to reassure you it’s correct
train_val_path = os.path.join(dataset_path, "VOC2012_train_val")
print(f"Path to train_val data: {train_val_path}")