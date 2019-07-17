/panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/env/hwr4_env/bin/python
./configs/warp.yaml
/panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/taylor_simple_hwr/utils.py:47: YAMLLoadWarning: calling yaml.load() without Loader=... is deprecated, as the default Loader is unsafe. Please read https://msg.pyyaml.org/load for full details.
  return fix_scientific_notation(yaml.load(stream))
Using config file ./configs/warp.yaml
{
  "TESTING": false,
  "SMALL_TRAINING": false,
  "plot_freq": 50,
  "load_path": false,
  "test_only": false,
  "experiment": "warp_only",
  "name": "warp",
  "output_folder": "./results",
  "epochs_to_run": 200,
  "save_freq": 5,
  "use_visdom": true,
  "debug": false,
  "training_jsons": [
    "prepare_IAM_Lines/raw_gts/lines/txt/training.json"
  ],
  "training_root": "../simple_hwr/data",
  "training_shuffle": true,
  "training_warp": true,
  "online_augmentation": false,
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
  "output_predictions": false,
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
  "scheduler_step": 100,
  "scheduler_gamma": 0.95,
  "training_suffle": false,
  "testing_suffle": false,
  "detach_embedding": false,
  "full_specs": "20190716_205203-warp_lr_0.0001_bs_8_warp_True_arch_False",
  "results_dir": "./results/warp_only/20190716_205203-warp_lr_0.0001_bs_8_warp_True_arch_False",
  "log_dir": "./results/warp_only/20190716_205203-warp_lr_0.0001_bs_8_warp_True_arch_False"
}
Number of training instances: 6161
Number of test instances: 1861 

Using GPU
Epoch: 1
updates: 50
updates: 100
updates: 150
updates: 200
updates: 250
updates: 300
updates: 350
updates: 400
updates: 450
updates: 500
updates: 550
updates: 600
updates: 650
updates: 700
updates: 750
Training CER: 1.0
Test CER: 1.0
Saving Best
Epoch: 2
updates: 800
updates: 850
updates: 900
updates: 950
updates: 1000
updates: 1050
updates: 1100
updates: 1150
updates: 1200
updates: 1250
updates: 1300
updates: 1350
updates: 1400
updates: 1450
updates: 1500
Training CER: 0.31601331915377695
Test CER: 0.3265656343406222
Saving Best
Epoch: 3
updates: 1550
updates: 1600
updates: 1650
updates: 1700
updates: 1750
updates: 1800
updates: 1850
updates: 1900
updates: 1950
updates: 2000
updates: 2050
updates: 2100
updates: 2150
updates: 2200
updates: 2250
updates: 2300
Training CER: 0.18046137775534365
Test CER: 0.18447851222711542
Saving Best
Epoch: 4
updates: 2350
updates: 2400
