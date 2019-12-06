from online_coordinate_parser import *
from pathlib import Path
import json
import cv2
import matplotlib.pyplot as plt
import sys
import pickle
from easydict import EasyDict as edict
from collections import defaultdict

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
    """ Create a list of dictionaries with the following keys:
                "full_img_path": str of path to image,
                "xml_path": str of path to XML,
                "image_path": str of path to image relative to project,
                "dataset": str training/test,
                "x": list of x coordinates, rescaled to be square with Y
                "y": list of y coordinates, normalized to 0-1
                "t": list of t coordinates, normalized to stroke length
                "start_times": start_times,
                "start_strokes": start_strokes,
                "x_to_y": ratio of x len to y len
                "raw": original stroke data from XML

        Export as JSON/Pickle
    """

    def __init__(self, max_strokes=3, square=True, output_folder_name='.',
                 xml_folder="prepare_online_data/line-level-xml/lineStrokes",
                 json_path="prepare_online_data/online_augmentation.json",
                 img_folder="prepare_online_data/lineImages",
                 data_folder="../../data",
                 render_images=True, test_set_size=None, combine_images=False):

        # Specify data folder:
            # xml, json, and default images relative to the data folder
            # output folder is also relative to the data folder

        self.project_root = Path("../..").resolve()
        self.absolute_data_folder = Path(data_folder).resolve()

        # Abbreviated data folder
        self.relative_data_folder = self.absolute_data_folder.relative_to(self.project_root)

        self.json_path = self.absolute_data_folder / json_path
        self.xml_folder = self.absolute_data_folder / xml_folder
        self.original_img_folder = img_folder # Needs to be relative path to data folder

        #if self.absolute_data_folder not in Path(output_folder).parents:
        current_folder = Path(".").resolve().name
        self.output_folder = self.absolute_data_folder / current_folder / output_folder_name
        print("Output:", self.output_folder.resolve())
        self.new_img_folder = (self.output_folder / "images").resolve()
        self.new_img_folder.mkdir(exist_ok=True, parents=True)
        self.data_dict = json.load(self.json_path.open("r"))
        self.output_dict = {"train": [], "test": []}
        self.max_strokes=max_strokes
        self.square=square
        self.render_images=render_images
        self.test_set_size = test_set_size
        self.combine_images = combine_images
        self.process_fn = self.process_multiple if self.combine_images else self.process_one


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
            if self.square and (x_to_y < .5 or x_to_y > 2):
                continue

            new_img_path = (self.new_img_folder / (file_name + f"_{i}")).with_suffix(".tif")

            new_item = {
                "full_img_path": item["image_path"],
                "xml_path": xml_path.resolve().relative_to(self.absolute_data_folder).as_posix(),
                "image_path": new_img_path.relative_to(self.absolute_data_folder).as_posix(),
                "dataset": dataset
            }
            new_item.update(sub_stroke_dict) # added to what's already in the substroke dictionary

            # Create images
            ratio = 1 if self.square else x_to_y

            if self.render_images:
                draw_strokes(normalize_stroke_list(sub_stroke_dict.raw), ratio, save_path=new_img_path, line_width=.8)
            new_items.append(new_item)

        ## Add shapes -- the system needs some time to actually perform the writing op before reading it back
        for item in new_items:
            new_img_path = self.absolute_data_folder / item["image_path"]
            shape = cv2.imread(new_img_path.as_posix()).shape
            item["shape"] = shape
            #ratio = item["x_to_y"]
            #print(shape, 61*ratio)
        return new_items

    @staticmethod
    def concat_substrokes(dict_list, last_point_remove=True):
        """

        Args:
            dict_list:
            last_point_remove (bool): Last point is a duplicate point for interpolation purposes, remove

        Returns:

        """
        #dict_keys(['x', 'y', 't', 'start_times', 'x_to_y', 'start_strokes', 'raw', 'tmin', 'tmax', 'trange'])

        spaces = {"x":.1, "time":.1}
        spaces["start_times"] = spaces["time"]
        spaces["t"] = spaces["time"]

        output_dict = edict() #defaultdict(list)

        ## X and Start Times and Strokes
        # 'start_strokes', 'raw'
        # Calculate tmax
        output_dict["tmax"] = 0
        output_dict.raw = []
        for _dict in dict_list:
            output_dict["tmax"] += _dict["tmax"]-_dict["tmin"]
            assert _dict["tmin"] == 0

            for key in ['x', 't', 'start_times']:
                ## Some of the lists have an extra item on the end
                last_index = -1 if last_point_remove and key != 'start_times' else None
                if key in output_dict:
                    output_dict[key] = np.append(output_dict[key] , (_dict[key][:last_index] + (np.max(output_dict[key]) + spaces[key])))
                else:
                    output_dict[key] = _dict[key]

            ## Don't add on to these
            for key in ['y', 'start_strokes']:
                last_index = -1 if last_point_remove and key != 'start_strokes' else None
                if key in output_dict:
                    output_dict[key] = np.append(output_dict[key] , (_dict[key][:last_index]))
                else:
                    output_dict[key] = _dict[key]

            ## Raw -
            output_dict.raw.append(_dict.raw) # This is now a list of "raw" lists, which are lists of strokes, where each stroke is a dict
            # i.e. data_instances->strokes->stroke_point_dict

            ## Do the same thing here, add the max item etc. ugh

        output_dict.x_to_y = np.max(output_dict.x) / np.max(output_dict.y)
        output_dict.trange = (0, output_dict.tmax)

        return output_dict


    @staticmethod
    # TODO fix the raw stuff (along with making sure the spaces are right OR do the combining on the RAW end of things obviously
    def process_multiple(items):
        """ Process multiple images to append together; no substrokes though

        Args:
            items:

        Returns:

        """

        # ONLY COMBINE IMAGES WITH OTHER IMAGES IN THE SAME CATEGORY, e.g. val1 with val1 etc.
        xml_paths = []
        meta_stroke_list = []
        file_names = []
        hyperparams = items
        for i, item in enumerate(items["data"]):
            file_names += Path(item["image_path"]).stem
            print(item)
            rel_path = Path(item["image_path"]).relative_to(hyperparams.original_img_folder).with_suffix(".xml")
            xml_path = hyperparams.xml_folder / rel_path

            # For each item, we can extract multiple stroke_lists by using a sliding
            # window across the image.  Thus multiple json items will point to the same
            # image, but different strokes within that image.
            stroke_list, _ = read_stroke_xml(xml_path)
            stroke_dict = prep_stroke_dict(stroke_list, time_interval=0, scale_time_distance=True) # list of dictionaries, 1 per file
            meta_stroke_list.append(stroke_dict)
            xml_paths.append(xml_path)
            if item["dataset"] != items["data"][i-1]["dataset"]: # make sure they are all the same dataset, val1, val2, train, test
                return
        dataset = item["dataset"]

        ## Merge two stroke dicts
        super_stroke_list = CreateDataset.concat_substrokes(meta_stroke_list)
        new_items = []

        x_to_y = super_stroke_list.x_to_y

        new_file_name = "_".join(file_names) + ".tif"
        new_img_path = hyperparams.new_img_folder / new_file_name

        new_item = {
            "full_img_path": item["image_path"], # full path of where the image is saved
            "xml_path": [xml_path.resolve().relative_to(hyperparams.absolute_data_folder).as_posix() for xml_path in xml_paths], # relative path to original XML file
            "image_path": new_img_path.relative_to(hyperparams.absolute_data_folder).as_posix(), # relative path
            "dataset": dataset
        }
        new_item.update(super_stroke_list) # added to what's already in the substroke dictionary

        # Create images
        if hyperparams.render_images:
            draw_strokes(normalize_stroke_list(super_stroke_list.raw[0]), x_to_y, save_path=new_img_path, line_width=.8)
        ## NEED TO FIX RAW, UGH
        new_items.append(new_item)

        ## Add shapes -- the system needs some time to actually perform the writing op before reading it back
        for item in new_items:
            new_img_path = hyperparams.absolute_data_folder / item["image_path"]
            shape = cv2.imread(new_img_path.as_posix()).shape
            item["shape"] = shape
            #ratio = item["x_to_y"]
            #print(shape, 61*ratio)
        return new_items

    @staticmethod
    def loop(temp_results, func=None):
        """ The non parallel version, better for error tracking

        Args:
            temp_results:
            func:

        Returns:

        """
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
                # If test set is specified to be smaller, add to training set after a certain size
                if item["dataset"] in ["train", "val1", "val2"] or (self.test_set_size and len(self.output_dict["test"]) > self.test_set_size):
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

        if self.combine_images:
            # for every item in the data dict, pick another item and combine them
            new_data_dict = []
            for i,d in enumerate(data_dict):
                new_data_dict.append({"data":((data_dict[i-1], data_dict[i]))})
            data_dict = new_data_dict

        ### Add all the hyperparameters to the item instead of keeping them in a class, seems to be faster
        hyper_param_dict = self.__dict__
        del hyper_param_dict["data_dict"]
        for i, item in enumerate(data_dict):
            item.update(hyper_param_dict)
            data_dict[i] = edict(item)

        if parallel:
            poolcount = multiprocessing.cpu_count()-1
            pool = multiprocessing.Pool(processes=poolcount)
            all_results = pool.imap_unordered(self.process_fn, tqdm(data_dict))  # iterates through everything all at once
            #all_results = pool.starmap(self.process_one, zip(tqdm(data_dict), hyper_param_dict))  # iterates through everything all at once

            pool.close()
        else:
            all_results = self.loop(tqdm(data_dict), func=self.process_fn)

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
    strokes = None      # None=MAX stroke
    square = False      # Don't require square images
    instances = None    # None=Use all available instances
    test_set_size = 500 # use leftover test images in Training
    combine_images = True # combine images to make them longer

    variant="TEST_AUGMENT"
    if square:
        variant += "Square"
    if instances is None:
        variant += "Full"
    else:
        variant += f"Small_{instances}"
    number_of_strokes = str(strokes) if isinstance(strokes, int) else "MAX"
    data_set = CreateDataset(max_strokes=strokes,
                             square=square,
                             output_folder_name=f"./{number_of_strokes}_stroke_v{variant}",
                             render_images=True,
                             test_set_size=test_set_size,
                             combine_images=combine_images)
    data_set.parallel(max_iter=instances, parallel=False)
