from pathlib import Path
import yaml

baseline_config = "./occlusion/online_only/online.yaml"

def create_configs():
    #
    with open(config, 'r') as stream:
        yaml.load(stream)

    for i in [.1,.4]:
        for k in [.4,.5]:



            with open(Path(output / 'RESUME.yaml'), 'w') as outfile:
                yaml.dump(export_config, outfile, default_flow_style=False, sort_keys=False)


if __name__=="__main__":


