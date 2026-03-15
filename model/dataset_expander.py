import os
import random
import shutil

SOURCE_DIR = r"C:\Users\Accer\Downloads\IMGUR5K-Handwriting-Dataset\IMGUR5K-Handwriting-Dataset\imgur5k-dataset"

TRAIN_DIR = r"C:\Users\Accer\Downloads\PlantDiseasePredictor\PlantDiseasePredictor\PlantDiseasePredictor\dataset\Binary_Plant_Non-Plant_Dataset\train\Not_Leaf"

VAL_DIR = r"C:\Users\Accer\Downloads\PlantDiseasePredictor\PlantDiseasePredictor\PlantDiseasePredictor\dataset\Binary_Plant_Non-Plant_Dataset\val\Not_Leaf"

TEST_DIR = r"C:\Users\Accer\Downloads\PlantDiseasePredictor\PlantDiseasePredictor\PlantDiseasePredictor\dataset\Binary_Plant_Non-Plant_Dataset\test\Not_Leaf"

TRAIN_COUNT = 300
VAL_COUNT = 80
TEST_COUNT = 50

valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

all_images = [
    f for f in os.listdir(SOURCE_DIR)
    if f.lower().endswith(valid_ext)
]

print(f"Total images available in source: {len(all_images)}")

required_total = TRAIN_COUNT + VAL_COUNT + TEST_COUNT

if len(all_images) < required_total:
    raise ValueError(
        f"Not enough images in source folder. Need {required_total}, found {len(all_images)}"
    )

random.shuffle(all_images)

def move_images(file_list, destination):

    os.makedirs(destination, exist_ok=True)

    for file_name in file_list:

        src = os.path.join(SOURCE_DIR, file_name)
        dst = os.path.join(destination, file_name)

        shutil.move(src, dst)

    print(f"Moved {len(file_list)} images to {destination}")

train_files = all_images[:TRAIN_COUNT]

val_files = all_images[
    TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT
]

test_files = all_images[
    TRAIN_COUNT + VAL_COUNT:
    TRAIN_COUNT + VAL_COUNT + TEST_COUNT
]

print("\nMoving TRAIN images...")
move_images(train_files, TRAIN_DIR)

print("\nMoving VAL images...")
move_images(val_files, VAL_DIR)

print("\nMoving TEST images...")
move_images(test_files, TEST_DIR)


print("\nDataset split completed safely.")
print(f"Train: {len(train_files)}")
print(f"Val: {len(val_files)}")
print(f"Test: {len(test_files)}")