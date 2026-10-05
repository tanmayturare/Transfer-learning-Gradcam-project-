import os

DATASET_DIR = "dataset"

folders = [
    "train/NORMAL",
    "train/PNEUMONIA",
    "validation/NORMAL",
    "validation/PNEUMONIA",
    "test/NORMAL",
    "test/PNEUMONIA"
]

print("Checking dataset structure...\n")

for folder in folders:

    path = os.path.join(DATASET_DIR, folder)

    if os.path.exists(path):

        files = os.listdir(path)

        print(f"{folder}")
        print(f"Number of files: {len(files)}")
        print()

    else:

        print(f"❌ Missing folder: {folder}")
        print()

print("Dataset structure check completed.")