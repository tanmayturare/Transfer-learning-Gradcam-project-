import cv2
import os

image_path = None

folders = [
    "dataset/train/NORMAL",
    "dataset/train/PNEUMONIA"
]

for folder in folders:

    if os.path.exists(folder):

        files = os.listdir(folder)

        for file in files:

            if file.lower().endswith((".jpg", ".jpeg", ".png")):

                image_path = os.path.join(folder, file)
                break

    if image_path:
        break


if image_path:

    print("Testing image:")
    print(image_path)

    image = cv2.imread(image_path)

    if image is not None:

        print("Image loaded successfully!")
        print("Image shape:", image.shape)

    else:

        print("ERROR: Image could not be loaded.")

else:

    print("No image found.")