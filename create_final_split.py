import os
import shutil
import random



SOURCE_TRAIN = "dataset/train"
SOURCE_VALIDATION = "dataset/validation"
SOURCE_TEST = "dataset/test"

OUTPUT_DIR = "dataset_final"

VALIDATION_RATIO = 0.10

random.seed(42)




for split in ["train", "validation", "test"]:
    for class_name in ["NORMAL", "PNEUMONIA"]:
        os.makedirs(
            os.path.join(OUTPUT_DIR, split, class_name),
            exist_ok=True
        )




IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")




for class_name in ["NORMAL", "PNEUMONIA"]:

    source_folder = os.path.join(SOURCE_TRAIN, class_name)

    images = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    random.shuffle(images)

    validation_count = int(len(images) * VALIDATION_RATIO)

    validation_images = images[:validation_count]
    train_images = images[validation_count:]

    print(f"\n{class_name}")
    print("Original training images :", len(images))
    print("New training images      :", len(train_images))
    print("New validation images    :", len(validation_images))

   
    for image in train_images:

        source = os.path.join(source_folder, image)
        destination = os.path.join(
            OUTPUT_DIR,
            "train",
            class_name,
            image
        )

        shutil.copy2(source, destination)

    
    for image in validation_images:

        source = os.path.join(source_folder, image)
        destination = os.path.join(
            OUTPUT_DIR,
            "validation",
            class_name,
            image
        )

        shutil.copy2(source, destination)




for class_name in ["NORMAL", "PNEUMONIA"]:

    source_folder = os.path.join(SOURCE_TEST, class_name)

    images = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    for image in images:

        source = os.path.join(source_folder, image)

        destination = os.path.join(
            OUTPUT_DIR,
            "test",
            class_name,
            image
        )

        shutil.copy2(source, destination)

    print(
        f"\nCopied {len(images)} test images for {class_name}"
    )




print("\n" + "=" * 50)
print("FINAL DATASET SPLIT")
print("=" * 50)

for split in ["train", "validation", "test"]:

    print(f"\n{split.upper()}")

    for class_name in ["NORMAL", "PNEUMONIA"]:

        folder = os.path.join(
            OUTPUT_DIR,
            split,
            class_name
        )

        count = len([
            f for f in os.listdir(folder)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ])

        print(f"{class_name}: {count}")

print("\nDataset creation completed.")