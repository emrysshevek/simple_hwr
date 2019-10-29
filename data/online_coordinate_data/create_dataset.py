from online_coordinate_parser import *
from pathlib import Path
import json
import cv2
import matplotlib.pyplot as plt
import sys
import pickle
from easydict import EasyDict as edict

sys.path.insert(0, "../../")
from hwr_utils.stroke_recovery import *
from hwr_utils.stroke_plotting import *
from tqdm import tqdm
import multiprocessing
from functools import wraps

normal_error_handling = False

# def error( *args, **kwargs):
#     print("ERROR!")
#     Stop
#     return "error"


def error_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        # Error handler does nothing
        if normal_error_handling:
            return func(*args, **kwargs)
        else:
            try:
                return func(*args, **kwargs)  # exits here if everything is working
            except Exception as e:
                return e

    setattr(sys.modules[func.__module__], func.__name__, wrapper)

    return wrapper


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

    #@error_handler
    #@staticmethod

    # def process_one(self, item):
    #     #the_dict = edict(self.__dict__)
    #     return self._process_one(self, item)

    # def process_one(self, item):
    #     return self.process_one(item)

    @staticmethod
    def process_one(item):
        self = item

        file_name = Path(item["image_path"]).stem
        rel_path = Path(item["image_path"]).relative_to(self.original_img_folder).with_suffix(".xml")
        xml_path = self.xml_folder / rel_path

        # For each item, we can extract multiple stroke_lists by using a sliding
        # window across the image.  Thus multiple json items will point to the same
        # image, but different strokes within that image.
        stroke_list, _ = read_stroke_xml(xml_path)
        stroke_dict = prep_stroke_dict(stroke_list, time_interval=0, scale_time_distance=True) # list of dictionaries, 1 per file
        all_substrokes = get_all_substrokes(stroke_dict, length=self.max_strokes) # subdivide file into smaller sets

        if item["dataset"] in ["test", "val1", "val2"]:
            dataset = "test"
        else:
            dataset = "train"

        new_items = []

        for i, sub_stroke_dict in enumerate(all_substrokes):
            x_to_y = sub_stroke_dict.x_to_y

            # Don't warp images too much
            if self.square and x_to_y < .5 or x_to_y > 2:
                continue

            img_path = (self.new_img_folder / (file_name + f"_{i}")).with_suffix(".tif")

            new_item = {
                "full_img_path": item["image_path"],
                "xml_path": xml_path.resolve().relative_to(self.data_folder).as_posix(),
                "image_path": img_path.relative_to(self.data_folder).as_posix(),
                "dataset": dataset,
                "gt": sub_stroke_dict,
                "stroke_list": sub_stroke_dict.raw,
                "x_to_y": sub_stroke_dict.x_to_y,
                "start_times": sub_stroke_dict.start_times.tolist()
            }

            # Create images
            ratio = 1 if self.square else x_to_y

            if self.render_images:
                draw_strokes(normalize_stroke_list(sub_stroke_dict.raw), ratio, save_path=img_path, line_width=.8)
            new_items.append(new_item)
        return new_items

    @staticmethod
    def loop(temp_results, func=None):
        all_results = []
        no_errors = True
        for i, result in enumerate(temp_results):
            if not func is None:
                result = func(result)
            if isinstance(result, Exception) and no_errors:
                print(result)
                no_errors = False
            else:
                all_results.append(result)
        return all_results

    def final_process(self, all_results):
        for d in all_results:
            for item in d:
                if item["dataset"]=="train":
                    self.output_dict["train"].append(item)
                elif item["dataset"] == "test":
                    self.output_dict["test"].append(item)

        # PICKLE IT
        pickle.dump(self.output_dict["train"], (self.output_folder / "train_online_coords.pickle").open("wb"))
        pickle.dump(self.output_dict["test"], (self.output_folder / "test_online_coords.pickle").open("wb"))

        # ALSO JSON
        self.prep_for_json(self.output_dict["train"])
        self.prep_for_json(self.output_dict["test"])

        print("Creating train_online_coords.json and test_online_coords.json...")
        json.dump(self.output_dict["train"], (self.output_folder / "train_online_coords.json").open("w"))
        json.dump(self.output_dict["test"], (self.output_folder / "test_online_coords.json").open("w"))

        return self.output_dict

    def parallel(self, max_iter=None, parallel=True):
        data_dict = self.data_dict
        if max_iter:
            data_dict = data_dict[:max_iter]

        ### Add all the hyperparameters to the item instead of keeping them in a class, seems to be faster
        hyper_param_dict = self.__dict__
        del hyper_param_dict["data_dict"]
        for i, item in enumerate(data_dict):
            item.update(hyper_param_dict)
            data_dict[i] = edict(item)

        if parallel:
            poolcount = multiprocessing.cpu_count()-1
            pool = multiprocessing.Pool(processes=poolcount)
            all_results = pool.imap_unordered(self.process_one, tqdm(data_dict))  # iterates through everything all at once
            #all_results = pool.starmap(self.process_one, zip(tqdm(data_dict), hyper_param_dict))  # iterates through everything all at once

            pool.close()
        else:
            all_results = self.loop(tqdm(data_dict), func=self.process_one)

        return self.final_process(all_results)

    def prep_for_json(self, iterable):
        if isinstance(iterable, list):
            for i in iterable:
                self.prep_for_json(i)
        elif isinstance(iterable, dict):
            for key, value in iterable.items():
                if isinstance(value, np.ndarray):
                    iterable[key] = value.tolist()
                else:
                    self.prep_for_json(value)
        elif isinstance(iterable, np.ndarray):
            pass

#def out_pickle(f)

if __name__ == "__main__":
    stroke = 3
    instances = 64
    data_set = CreateDataset(max_strokes=stroke, square=True, instances=instances,
                                 output_folder=f"./{stroke}_stroke_{instances}_v2", render_images=True)
    data_set.parallel(max_iter=None, parallel=True)
