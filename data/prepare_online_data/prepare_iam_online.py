from xml.etree import ElementTree as ET
import json
import os

xml_dir = 'original-xml-all'
img_dir = 'cropped_images'

img_json = []
char_set = set()

xml_files = set()
for root, dirs, files in os.walk(xml_dir):
    xml_files.update([os.path.join(root, file) for file in files])

for file in xml_files:
    root = ET.parse(file).getroot()
    transcription = root.find('Transcription')

    if not transcription:
        continue

    for line in transcription.findall('TextLine'):
        gt = line.get('text')
        img_path = os.path.join('prepare_online_data', img_dir, line.get('id') + '.tif')
        if os.path.exists(os.path.join(os.getcwd(), img_dir, line.get('id') + '.tif')):
            print(gt, img_path)
            img_json.append({'gt': gt, 'image_path': img_path, 'augmentation': True})
            char_set.update(gt)

char_set = {'char_to_idx': {char: i+1 for i, char in enumerate(list(char_set))}, 'idx_to_char': {str(i+1): char for i, char in enumerate(list(char_set))}}
print()
print(char_set)

with open('online_char_set.json', 'w') as fp:
    json.dump(char_set, fp, indent=2)

with open('online_augmentation.json', 'w') as fp:
    json.dump(img_json, fp, indent=2)