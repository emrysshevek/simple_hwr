from online_coordinate_parser import *
from online_coordinate_parser import *
from pathlib import Path
import json
import cv2
import matplotlib.pyplot as plt
from hwr_utils.stroke_recovery import *

def create_dataset(max_strokes=3, square=True, instances=50, output_folder='.',
                   xml_folder="../prepare_online_data/line-level-xml",
                   json_path="../prepare_online_data/online_augmentation.json",
                   img_folder="prepare_online_data/lineImages", render_images=True):
    # Loop through files
    # Find image paths
    # create dicitonary
    json_path = Path(json_path)
    xml_folder = Path(xml_folder)
    original_img_folder = Path(img_folder)
    output_folder = Path(output_folder)
    new_img_folder = (Path(output_folder) / "images").resolve()
    data_folder = Path("..").resolve()
    new_img_folder.mkdir(exist_ok=True, parents=True)
    data_dict = json.load(json_path.open("r"))

    output_dict = {"train": [], "test": []}

    for item in data_dict:
        file_name = Path(item["image_path"]).name
        rel_path = Path(item["image_path"]).relative_to(original_img_folder).with_suffix(".xml")
        xml_path = xml_folder / rel_path
        gts, stroke_list, x_to_y = get_gts(xml_path, instances=instances, max_stroke_count=max_strokes)

        if item["dataset"] in ["test", "val1", "val2"]:
            dataset = "test"
        else:
            dataset = "train"

        img_path = (new_img_folder / file_name)
        new_item = {
            "full_img_path": item["image_path"],
            "xml_path": xml_path.resolve().relative_to(data_folder).as_posix(),
            "image_path": img_path.relative_to(data_folder).as_posix(),
            "dataset": dataset,
            "gt": [gt.tolist() for gt in gts],
            "stroke_list": stroke_list,
            "x_to_y": x_to_y
        }

        # Create images
        ratio = 1 if square else x_to_y

        # Don't warp images too much
        if square and x_to_y < .5 or x_to_y > 2:
            continue
        if render_images:
            draw_strokes(normalize_stroke_list(stroke_list), ratio, save_path=img_path)
        output_dict[dataset].append(new_item)

    json.dump(output_dict["train"], (output_folder / "train_online_coords.json").open("w"), indent=2)
    json.dump(output_dict["test"], (output_folder / "test_online_coords.json").open("w"), indent=2)

    return output_dict


if __name__ == "__main__":
    stroke = 3
    instances = 16
    output_dict = create_dataset(max_strokes=stroke, square=True, instances=instances,
                                 output_folder=f"./{stroke}_stroke_{instances}", render_images=False)
