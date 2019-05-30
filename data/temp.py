import json
import os


json_path = '/home/mason/fslg_hwr/simple_hwr/data/prepare_online_data/train_augmentation.json'
image_dir = 'prepare_online_data/cropped_images'

with open(json_path) as fp:
    data = json.load(fp)

# for x in data:
#     x['image_path'] = os.path.join(image_dir, x['image_path'])

for x in data:
    del x['augmentation']

with open(json_path, 'w') as fp:
    json.dump(data, fp, indent=2)
