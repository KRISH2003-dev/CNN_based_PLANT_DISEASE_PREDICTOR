import os
import random
import shutil

SOURCE_ROOT = r"C:\Users\Accer\Downloads\archive (2)\image"

TRAIN_DIR = r"C:\Users\Accer\Downloads\PlantDiseasePredictor\PlantDiseasePredictor\PlantDiseasePredictor\dataset\Binary_Plant_Non-Plant_Dataset\train\Not_Leaf"

VAL_DIR = r"C:\Users\Accer\Downloads\PlantDiseasePredictor\PlantDiseasePredictor\PlantDiseasePredictor\dataset\Binary_Plant_Non-Plant_Dataset\val\Not_Leaf"

TEST_DIR = r"C:\Users\Accer\Downloads\PlantDiseasePredictor\PlantDiseasePredictor\PlantDiseasePredictor\dataset\Binary_Plant_Non-Plant_Dataset\test\Not_Leaf"

random.seed(42)

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

subfolders = [
    f for f in os.listdir(SOURCE_ROOT)
    if os.path.isdir(os.path.join(SOURCE_ROOT, f))
]

print(f"Found {len(subfolders)} folders to process.\n")

total_train = 0
total_val = 0
total_test = 0

for folder in subfolders:

    folder_path = os.path.join(SOURCE_ROOT, folder)

    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(VALID_EXTENSIONS)
    ]

    if len(images) == 0:
        print(f"Skipping empty folder: {folder}")
        continue

    random.shuffle(images)

    total = len(images)

    train_count = int(total * 0.70)
    val_count = int(total * 0.20)
    test_count = total - train_count - val_count

    train_files = images[:train_count]
    val_files = images[train_count:train_count + val_count]
    test_files = images[train_count + val_count:]

    print(f"\nProcessing folder: {folder}")
    print(f"Total images: {total}")
    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    for file in train_files:

        src = os.path.join(folder_path, file)
        dst = os.path.join(TRAIN_DIR, f"{folder}_{file}")

        shutil.move(src, dst)

    total_train += len(train_files)

    for file in val_files:

        src = os.path.join(folder_path, file)
        dst = os.path.join(VAL_DIR, f"{folder}_{file}")

        shutil.move(src, dst)

    total_val += len(val_files)

    for file in test_files:

        src = os.path.join(folder_path, file)
        dst = os.path.join(TEST_DIR, f"{folder}_{file}")

        shutil.move(src, dst)

    total_test += len(test_files)

print("\n-----------------------------------")
print("Dataset splitting completed safely")
print("-----------------------------------")

print(f"Total moved to TRAIN: {total_train}")
print(f"Total moved to VAL:   {total_val}")
print(f"Total moved to TEST:  {total_test}")