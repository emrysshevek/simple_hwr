from online_coordinate_parser import *
from pathlib import Path
import json
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "../../")
from hwr_utils.stroke_recovery import *
from tqdm import tqdm
import multiprocessing

class CreateDataset:

    def __init__(self, max_strokes=3, square=True, instances=50, output_folder='.',
                   xml_folder="../prepare_online_data/line-level-xml/lineStrokes",
                   json_path="../prepare_online_data/online_augmentation.json",
                   img_folder="prepare_online_data/lineImages", render_images=True):
        # Loop through files
        # Find image paths
        # create dicitonary
        self.json_path = Path(json_path)
        self.xml_folder = Path(xml_folder)
        self.original_img_folder = Path(img_folder)
        self.output_folder = Path(output_folder)
        self.new_img_folder = (Path(output_folder) / "images").resolve()
        self.data_folder = Path("..").resolve()
        self.new_img_folder.mkdir(exist_ok=True, parents=True)
        self.data_dict = json.load(self.json_path.open("r"))
        self.output_dict = {"train": [], "test": []}
        self.max_strokes=max_strokes
        self.square=square
        self.instances=instances
        self.render_images=render_images

    def process_one(self, item):
        file_name = Path(item["image_path"]).stem
        rel_path = Path(item["image_path"]).relative_to(self.original_img_folder).with_suffix(".xml")
        xml_path = self.xml_folder / rel_path

        # For each item, we can extract multiple stroke_lists by using a sliding
        # window across the image.  Thus multiple json items will point to the same
        # image, but different strokes within that image.
        stroke_list, _ = read_stroke_xml(xml_path)
        stroke_dict = prep_stroke_dict(stroke_list, time_interval=0, scale_time_distance=True)
        all_substrokes = get_all_substrokes(stroke_dict, length=self.max_strokes)

        if item["dataset"] in ["test", "val1", "val2"]:
            dataset = "test"
        else:
            dataset = "train"

        new_items = []
        for sub_stroke_dict in all_substrokes:
            img_path = (self.new_img_folder / (file_name + f"_{i}")).with_suffix(".tif")

            new_item = {
                "full_img_path": item["image_path"],
                "xml_path": xml_path.resolve().relative_to(self.data_folder).as_posix(),
                "image_path": img_path.relative_to(self.data_folder).as_posix(),
                "dataset": dataset,
                "gt": [gt.tolist() for gt in gts],
                "stroke_list": stroke_list,
                "x_to_y": x_to_y
            }

            # Create images
            ratio = 1 if self.square else x_to_y

            # Don't warp images too much
            if self.square and x_to_y < .5 or x_to_y > 2:
                continue
            if self.render_images:
                draw_strokes(normalize_stroke_list(stroke_list), ratio, save_path=img_path, line_width=.8)
            new_items.append(new_item)
        return new_items


    def parallel(self):
        data_dict = self.data_dict
        all_results = []
        poolcount = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=poolcount)

        #data_dict = data_dict[:30]
        temp_results = pool.imap_unordered(self.process_one, tqdm(data_dict))  # iterates through everything all at once
        pool.close()

        # Loop through results as they come in
        for result in temp_results:
            all_results.append(result)

        for d in all_results:
            for item in d:
                if item["dataset"]=="train":
                    self.output_dict["train"].append(item)
                elif item["dataset"] == "test":
                    self.output_dict["test"].append(item)

        print("Creating train_online_coords.json and test_online_coords.json...")
        json.dump(self.output_dict["train"], (self.output_folder / "train_online_coords.json").open("w"), indent=2)
        json.dump(self.output_dict["test"], (self.output_folder / "test_online_coords.json").open("w"), indent=2)

        return self.output_dict

if __name__ == "__main__":
    stroke = 3
    instances = 64
    data_set = CreateDataset(max_strokes=stroke, square=True, instances=instances,
                                 output_folder=f"./{stroke}_stroke_{instances}_v2", render_images=True)
    data_set.parallel()