import os

DATASET_DIR = "dataset"

extensions = (".jpg", ".jpeg", ".png")

folders = [
    "train/NORMAL",
    "train/PNEUMONIA",
    "validation/NORMAL",
    "validation/PNEUMONIA",
    "test/NORMAL",
    "test/PNEUMONIA"
]

print("Checking image files...\n")

for folder in folders:

    path = os.path.join(DATASET_DIR, folder)

    if not os.path.exists(path):
        print(f"Missing: {folder}")
        continue

    files = os.listdir(path)

    image_files = [
        f for f in files
        if f.lower().endswith(extensions)
    ]

    print(
        f"{folder}: "
        f"{len(image_files)} image files"
    )

print("\nImage inspection completed.")