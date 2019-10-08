from online_coordinate_parser import *

### TO DO:
    # Create dataset
    # Get attention to work
    # Render solution

    # Add tokens and adjust loss
    # Rescale -- probably no activation on last layer

def create_one_stroke_dataset(xml_folder, json="../prepare_online_data/online_augmentation.json"):
    # Loop through files
    # Find image paths
    # create dicitonary

    data_dict = json.load(json)

    gts = get_gts(path, instances=30, max_stroke_count=2)


if __name__=="__main__":
