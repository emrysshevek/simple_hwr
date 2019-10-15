import sys
from xml.etree import ElementTree as ET
import json
from tqdm import tqdm
from os import getcwd, walk
from os.path import join, normpath, basename, exists
from html import unescape

def clean_text(s):
    return unescape(s)

def strip_rightmost_alphabetic(s):
    L = list(s)
    for i in range(len(s)):
        if L[-1].isalpha():
            L.pop()
        else:
            break
    return "".join(L)

def get_image_path_from_id(img_id):
    components = img_id.split("-")
    folder_1 = components[0]
    folder_2 = "-".join([components[0], strip_rightmost_alphabetic(components[1])])
    fname = img_id + ".tif"
    return join(IMG_DIR, folder_1, folder_2, fname)


def prepend_cwd(img_path):
    cwd = basename(normpath(getcwd()))
    return join(cwd, img_path)

def get_xml_files(xml_dir):
    xml_files = []
    for root, dirs, files in walk(xml_dir):
        xml_files += [join(root, file) for file in files]
    return xml_files


if __name__ == "__main__":
    IMG_DIR = 'lineImages'
    XML_DIR = 'original-xml-all'
    img_json = []

    if not exists(IMG_DIR) or not exists(XML_DIR):
        raise Exception(f"Verify {prepend_cwd(IMG_DIR)} and {prepend_cwd(XML_DIR)} exist.")

    for file in tqdm(get_xml_files(XML_DIR)):
        root = ET.parse(file).getroot()
        transcription = root.find('Transcription')

        if not transcription:
            continue

        for line in transcription.findall('TextLine'):
            gt = clean_text(line.get('text'))
            img_id = line.get('id')
            img_path = get_image_path_from_id(img_id)
            writer_id = root.find('General').find('Form').attrib["writerID"]
            if exists(img_path):
                full_img_path = prepend_cwd(img_path)
                img_json.append({
                    'gt': gt,
                    'image_path': full_img_path,
                    'online': True,
                    'writer_id': writer_id
                })

    if img_json:
        with open('online_augmentation.json', 'w') as fp:
            json.dump(img_json, fp, indent=2)
    else:
        raise Exception("No images found!")
