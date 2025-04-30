import os
import random

# Scene categories and their counts
scene_categories = {
    'chess': 6,
    'fire': 4,
    'heads': 2,
    'office': 10,
    'pumpkin': 8,
    'redkitchen': 14,
    'stairs': 6
}

base_dir = '/data/kmirakho/l3d_proj/7scenes_processed'

# Set random seed for reproducibility
random.seed(42)

train_scenes = []
test_scenes = []

for category, count in scene_categories.items():
    scenes = [os.path.join(base_dir, f"{category}/{str(i).zfill(2)}") for i in range(1, count + 1)]
    random.shuffle(scenes)
    split_idx = int(0.8 * count)
    train_scenes.extend(scenes[:split_idx])
    test_scenes.extend(scenes[split_idx:])

with open('split_7scenes.txt', 'w') as f:
    f.write(f"Train scenes:\n")
    for scene in train_scenes:
        f.write(scene + '\n')
    f.write(f"\nTest scenes:\n")
    for scene in test_scenes:
        f.write(scene + '\n')

print(f"Train scenes ({len(train_scenes)})")
print(f"Test scenes ({len(test_scenes)})")
