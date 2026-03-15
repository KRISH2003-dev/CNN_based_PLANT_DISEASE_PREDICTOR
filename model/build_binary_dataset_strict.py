import os
import random
import shutil
from pathlib import Path

SEED = 42
random.seed(SEED)

PLANT_DATASET_PATH = "dataset/New Plant Diseases Dataset(Augmented)/valid"
COCO_DATASET_PATH = r"C:\Users\Accer\Downloads\val2017\val2017"
TARGET_ROOT = "dataset/Binary_Plant_Non-Plant_Dataset"

TOTAL_PER_CLASS = 5000
PER_CLASS_CORE = 120

TRAIN_COUNT = 3500
VALID_COUNT = 1000
TEST_COUNT = 500


def verify_structure():
    required_paths = [
        f"{TARGET_ROOT}/train/Leaf",
        f"{TARGET_ROOT}/train/Not_Leaf",
        f"{TARGET_ROOT}/valid/Leaf",
        f"{TARGET_ROOT}/valid/Not_Leaf",
        f"{TARGET_ROOT}/test/Leaf",
        f"{TARGET_ROOT}/test/Not_Leaf",
    ]

    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required folder: {path}")

    print("Folder structure verified.")


def clear_existing_files():
    for split in ["train", "valid", "test"]:
        for cls in ["Leaf", "Not_Leaf"]:
            folder = Path(TARGET_ROOT) / split / cls
            for file in folder.iterdir():
                if file.is_file():
                    file.unlink()

    print("Old files cleared from target folders.")

def sample_leaf_images():
    plant_root = Path(PLANT_DATASET_PATH)
    class_folders = [d for d in plant_root.iterdir() if d.is_dir()]

    all_selected = []
    remaining_pool = []

    print(f"Detected {len(class_folders)} disease classes.")

    for class_folder in class_folders:
        images = list(class_folder.glob("*"))

        if len(images) < PER_CLASS_CORE:
            raise ValueError(
                f"Class {class_folder.name} has fewer than {PER_CLASS_CORE} images."
            )

        sampled_core = random.sample(images, PER_CLASS_CORE)
        all_selected.extend(sampled_core)

        leftovers = list(set(images) - set(sampled_core))
        remaining_pool.extend(leftovers)

    print(f"Core balanced images collected: {len(all_selected)}")

    remaining_needed = TOTAL_PER_CLASS - len(all_selected)

    if remaining_needed > 0:
        if len(remaining_pool) < remaining_needed:
            raise ValueError("Not enough images to complete 5000 total.")
        extra = random.sample(remaining_pool, remaining_needed)
        all_selected.extend(extra)

    print(f"Total Leaf images selected: {len(all_selected)}")

    return all_selected

def sample_not_leaf_images():
    coco_root = Path(COCO_DATASET_PATH)
    all_coco_images = list(coco_root.glob("*"))

    if len(all_coco_images) < TOTAL_PER_CLASS:
        raise ValueError("Not enough COCO images available.")

    sampled = random.sample(all_coco_images, TOTAL_PER_CLASS)

    print(f"Total Not_Leaf images selected: {len(sampled)}")
    return sampled

def split_and_copy(images, class_name):
    random.shuffle(images)

    train_split = images[:TRAIN_COUNT]
    valid_split = images[TRAIN_COUNT:TRAIN_COUNT + VALID_COUNT]
    test_split = images[TRAIN_COUNT + VALID_COUNT:]

    for img in train_split:
        shutil.copy(img, f"{TARGET_ROOT}/train/{class_name}")

    for img in valid_split:
        shutil.copy(img, f"{TARGET_ROOT}/valid/{class_name}")

    for img in test_split:
        shutil.copy(img, f"{TARGET_ROOT}/test/{class_name}")

def main():
    verify_structure()
    clear_existing_files()

    print("\nSampling Leaf images (Stratified)...")
    leaf_images = sample_leaf_images()

    print("\nSampling Not_Leaf images...")
    not_leaf_images = sample_not_leaf_images()

    print("\nCopying Leaf splits...")
    split_and_copy(leaf_images, "Leaf")

    print("Copying Not_Leaf splits...")
    split_and_copy(not_leaf_images, "Not_Leaf")

    print("\nDataset successfully built.")
    print("Per class distribution:")
    print(f"Train: {TRAIN_COUNT}")
    print(f"Valid: {VALID_COUNT}")
    print(f"Test : {TEST_COUNT}")


if __name__ == "__main__":
    main()