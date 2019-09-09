from __future__ import print_function

import sys
from train import *

import matplotlib
matplotlib.use('Agg')

VALIDATION_PATH = 'prepare_online_data/test_augmentation.json'


def main():
    config_path = sys.argv[1]

    with open(config_path) as f:
        config = json.load(f)

    results_dir = os.path.join('results', config['name'])
    if len(results_dir) > 0 and not os.path.exists(results_dir):
        os.makedirs(results_dir)

    train_config = config['train_data']
    test_config = config['test_data']

    char_to_idx, idx_to_char, char_freq = character_set.make_char_set(train_config['paths'], root=train_config['root'])

    train_dataloader, test_dataloader = make_dataloaders(
        train_config['paths'], train_config['root'], [VALIDATION_PATH], train_config['root'], char_to_idx,
        config['network']['input_height'], config['warp'], shuffle_train=train_config['shuffle'], shuffle_test=test_config['shuffle']
    )

    n_train_instances = len(train_dataloader.dataset)

    hw = crnn.create_CRNN({
        'cnn_out_size': config['network']['cnn_out_size'],
        'num_of_channels': 3,
        'num_of_outputs': len(idx_to_char) + 1
    })

    if torch.cuda.is_available():
        hw.cuda()
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    model_path = os.path.join('results', config['name'], config['name']+"_model.pt")

    hw.load_state_dict(torch.load(model_path))

    test_loss = test(hw, test_dataloader, idx_to_char, dtype)
    print(test_loss)


if __name__ == "__main__":
    main()