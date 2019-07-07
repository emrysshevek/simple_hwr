/panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/env/hwr4_env/bin/python
./configs/baseline.yaml
/panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/taylor_simple_hwr/utils.py:44: YAMLLoadWarning: calling yaml.load() without Loader=... is deprecated, as the default Loader is unsafe. Please read https://msg.pyyaml.org/load for full details.
  return fix_scientific_notation(yaml.load(stream))
./results/baseline
./results/baseline/20190707_113807-warp_augmented_lr_0.0001_bs_8_warp_True_arch_False
./results/baseline/20190707_113807-warp_augmented_lr_0.0001_bs_8_warp_True_arch_False
Using config file ./configs/baseline.yaml
{
  "TESTING": false,
  "load_path": false,
  "test_only": false,
  "experiment": "baseline",
  "name": "warp_augmented",
  "output_folder": "./results",
  "epochs_to_run": 200,
  "save_freq": 5,
  "use_visdom": true,
  "debug": false,
  "training_jsons": [
    "prepare_IAM_Lines/raw_gts/lines/txt/training.json",
    "prepare_online_data/online_augmentation.json"
  ],
  "training_root": "../simple_hwr/data",
  "training_shuffle": false,
  "training_warp": true,
  "online_augmentation": true,
  "writer_id_pickles": [
    "prepare_IAM_Lines/writer_IDs.pickle",
    "prepare_online_data/writer_IDs.pickle"
  ],
  "testing_jsons": [
    "prepare_IAM_Lines/raw_gts/lines/txt/test.json"
  ],
  "testing_root": "../simple_hwr/data",
  "testing_shuffle": false,
  "testing_warp": false,
  "output_predictions": true,
  "style_encoder": false,
  "batch_size": 8,
  "input_height": 60,
  "cnn_out_size": 1024,
  "num_of_channels": 3,
  "rnn_dimension": 512,
  "recognizer_dropout": 0.5,
  "writer_dropout": 0.4,
  "writer_rnn_dimension": 128,
  "writer_rnn_output_size": 256,
  "mlp_layers": [
    64,
    "embedding",
    64
  ],
  "embedding_size": 32,
  "learning_rate": 0.0001,
  "scheduler_step": 1500,
  "scheduler_gamma": 0.9,
  "full_specs": "20190707_113807-warp_augmented_lr_0.0001_bs_8_warp_True_arch_False",
  "results_dir": "./results/baseline/20190707_113807-warp_augmented_lr_0.0001_bs_8_warp_True_arch_False",
  "log_dir": "./results/baseline/20190707_113807-warp_augmented_lr_0.0001_bs_8_warp_True_arch_False"
}
Number of training instances: 11001
Number of test instances: 1861 

Using GPU
Epoch  1
50
100
150
200
250
300
350
400
450
500
550
600
650
700
750
800
