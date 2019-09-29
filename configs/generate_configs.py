import warnings
from pathlib import Path
import yaml
from itertools import product

baseline_configs = "./occlusion/online_or_offline_only/warp/offline.yaml", "./occlusion/online_or_offline_only/warp/online.yaml"
baseline_configs = "./occlusion/online_or_offline_only/offline.yaml", "./occlusion/online_or_offline_only/online.yaml"
baseline_configs = "./occlusion/gaussian_noise/offline.yaml", "./occlusion/gaussian_noise/online.yaml"
baseline_configs = "./MUNIT/iamlong.yaml", "./MUNIT/munit+iamlong.yaml", "./MUNIT/munit.yaml"


# "training_blur_level": 1.5,
#                 "training_random_distortions": False,
#                 "training_distortion_sigma": 6.0,
#                 "testing_blur": False,


variation_dict = {"training_random_distortions": [True, False], "training_blur": [True, False]} # randomly choose freq at most this; randomly choose level, at most this
baseline_dict = {"occlusion_level": 0}
baseline_dict = False

def cartesian_product(inp):
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))

def check_yaml(yaml_dict):
    for key, value in yaml_dict.items():
        if value == 'null':
            warnings.warn("Detected 'null' as string; changing to None")
            yaml_dict[key] = None
        elif isinstance(value, str):
            yaml_dict[key] = get_int(value)
        elif isinstance(value, list):
            new_list = []
            for v in value:
                new_list.append(get_int(v))
            yaml_dict[key] = new_list
    return yaml_dict

def get_int(s):
    if isinstance(s, str):
        if s.lower() == 'true':
            return True
        elif s.lower() == 'false':
            return False
        else:
            try:
                return int(s)
            except ValueError:
                try:
                    return float(s)
                except ValueError:
                    return s
    else:
        return s
def replace_config(yaml_path, variation_list, new_folder="variants"):
    yaml_path = Path(yaml_path)
    yaml_file = check_yaml(yaml.load(Path(yaml_path).read_text(), Loader=yaml.SafeLoader))
    name = yaml_path.stem
    parent = yaml_path.parent
    ext = yaml_path.suffix

    output_dir = Path(parent / new_folder).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    output_files = []
    for variant in variation_list:
        new_yaml_file = yaml_file.copy()

        output_file = name
        for key, value in variant.items():
            new_yaml_file[key] = value
            output_file += f"_{value}"

        #with open((parent / name).with_suffix(ext), "w") as f:
        with (output_dir / (output_file + ext)).open(mode='w') as f:
            yaml.dump(new_yaml_file, f, default_flow_style=False, sort_keys=False)

        output_files.append(output_file)
    return output_files


def main(baseline_file, variation_dict, baseline_dict=None):
    all_combinations = list(cartesian_product(variation_dict))

    if baseline_dict:
        all_combinations += [baseline_dict]
        print(all_combinations)
    all_files = replace_config(baseline_file, all_combinations)
    print(all_files)

if __name__=="__main__":
    for config in baseline_configs:
        main(config, variation_dict, baseline_dict)