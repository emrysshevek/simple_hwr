/panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/env/hwr4_env/bin/python
./configs/baseline.yaml
/panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/taylor_simple_hwr/utils.py:47: YAMLLoadWarning: calling yaml.load() without Loader=... is deprecated, as the default Loader is unsafe. Please read https://msg.pyyaml.org/load for full details.
  return fix_scientific_notation(yaml.load(stream))
Using config file ./configs/baseline.yaml
{
  "TESTING": false,
  "SMALL_TRAINING": false,
  "plot_freq": 50,
  "load_path": false,
  "test_only": false,
  "experiment": "baseline",
  "name": "true_baseline",
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
  "training_warp": false,
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
  "output_predictions": true,
  "style_encoder": false,
  "batch_size": 16,
  "input_height": 60,
  "cnn_out_size": 1024,
  "num_of_channels": 1,
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
  "training_suffle": false,
  "testing_suffle": false,
  "detach_embedding": false,
  "full_specs": "20190713_165320-true_baseline_lr_0.0001_bs_16_warp_False_arch_False",
  "results_dir": "./results/baseline/20190713_165320-true_baseline_lr_0.0001_bs_16_warp_False_arch_False",
  "log_dir": "./results/baseline/20190713_165320-true_baseline_lr_0.0001_bs_16_warp_False_arch_False"
}
Number of training instances: 6161
Number of test instances: 1861 

Using GPU
Epoch  1
Instances: 50
Instances: 100
Instances: 150
Instances: 200
Instances: 250
Instances: 300
Instances: 350
Training CER 1.013911421002099
Test CER 1.0
Saving Best
Epoch  2
Instances: 400
Instances: 450
Instances: 500
Instances: 550
Instances: 600
Instances: 650
Instances: 700
Instances: 750
Training CER 1.0
Test CER 1.0
Epoch  3
Instances: 800
Instances: 850
Instances: 900
Instances: 950
Instances: 1000
Instances: 1050
Instances: 1100
Instances: 1150
Training CER 0.7793735721942313
Test CER 0.3538436313931618
Saving Best
Epoch  4
Instances: 1200
Instances: 1250
Instances: 1300
Instances: 1350
Instances: 1400
Instances: 1450
Instances: 1500
Training CER 0.2235352038000469
Test CER 0.24076169561287664
Saving Best
Epoch  5
Instances: 1550
Instances: 1600
Instances: 1650
Instances: 1700
Instances: 1750
Instances: 1800
Instances: 1850
Instances: 1900
Training CER 0.13879987103761504
Test CER 0.19361176634337504
Saving Best
Saving Best
Epoch  6
Instances: 1950
Instances: 2000
Instances: 2050
Instances: 2100
Instances: 2150
Instances: 2200
Instances: 2250
Instances: 2300
Training CER 0.1014609618140971
Test CER 0.14635116337179868
Saving Best
Epoch  7
Instances: 2350
Instances: 2400
Instances: 2450
Instances: 2500
Instances: 2550
Instances: 2600
Instances: 2650
Instances: 2700
Training CER 0.07755408985067919
Test CER 0.150208314917152
Epoch  8
Instances: 2750
Instances: 2800
Instances: 2850
Instances: 2900
Instances: 2950
Instances: 3000
Instances: 3050
Training CER 0.05790095908202171
Test CER 0.13190321042858136
Saving Best
Epoch  9
Instances: 3100
Instances: 3150
Instances: 3200
Instances: 3250
Instances: 3300
Instances: 3350
Instances: 3400
Instances: 3450
Training CER 0.043966789709867456
Test CER 0.1378851576436314
Epoch  10
Instances: 3500
Instances: 3550
Instances: 3600
Instances: 3650
Instances: 3700
Instances: 3750
Instances: 3800
Instances: 3850
Training CER 0.03403354823009007
Test CER 0.12982341737560887
Saving Best
Saving Best
Epoch  11
Instances: 3900
Instances: 3950
Instances: 4000
Instances: 4050
Instances: 4100
Instances: 4150
Instances: 4200
Training CER 0.024884061243025606
Test CER 0.14112487514489105
Epoch  12
Instances: 4250
Instances: 4300
Instances: 4350
Instances: 4400
Instances: 4450
Instances: 4500
Instances: 4550
Instances: 4600
Training CER 0.019103449235248482
Test CER 0.12706747652304884
Saving Best
Epoch  13
Instances: 4650
Instances: 4700
Instances: 4750
Instances: 4800
Instances: 4850
Instances: 4900
Instances: 4950
Instances: 5000
Training CER 0.016961602466574518
Test CER 0.1304690262996521
Epoch  14
Instances: 5050
Instances: 5100
Instances: 5150
Instances: 5200
Instances: 5250
Instances: 5300
Instances: 5350
Instances: 5400
Training CER 0.01502197639506318
Test CER 0.13674614836939328
Epoch  15
Instances: 5450
Instances: 5500
Instances: 5550
Instances: 5600
Instances: 5650
Instances: 5700
Instances: 5750
Training CER 0.012870575143800224
Test CER 0.12835581812869937
Saving Best
Epoch  16
Instances: 5800
Instances: 5850
Instances: 5900
Instances: 5950
Instances: 6000
Instances: 6050
Instances: 6100
Instances: 6150
Training CER 0.01016031769850237
Test CER 0.1264298094776017
Saving Best
Epoch  17
Instances: 6200
Instances: 6250
Instances: 6300
Instances: 6350
Instances: 6400
Instances: 6450
Instances: 6500
Instances: 6550
Training CER 0.011515391892148417
Test CER 0.12733222164931307
Epoch  18
Instances: 6600
Instances: 6650
Instances: 6700
Instances: 6750
Instances: 6800
Instances: 6850
Instances: 6900
Training CER 0.011042629912635892
Test CER 0.122137188980028
Saving Best
Epoch  19
Instances: 6950
Instances: 7000
Instances: 7050
Instances: 7100
Instances: 7150
Instances: 7200
Instances: 7250
Instances: 7300
Training CER 0.008662550848881742
Test CER 0.12603079860394514
Epoch  20
Instances: 7350
Instances: 7400
Instances: 7450
Instances: 7500
Instances: 7550
Instances: 7600
Instances: 7650
Instances: 7700
Training CER 0.008677957537720838
Test CER 0.1320786758676924
Saving Best
Epoch  21
Instances: 7750
Instances: 7800
Instances: 7850
Instances: 7900
Instances: 7950
Instances: 8000
Instances: 8050
Instances: 8100
Training CER 0.009328663446022019
Test CER 0.12831404860165316
Epoch  22
Instances: 8150
Instances: 8200
Instances: 8250
Instances: 8300
Instances: 8350
Instances: 8400
Instances: 8450
Training CER 0.0060974988410571595
Test CER 0.12210971500900278
Saving Best
Epoch  23
Instances: 8500
Instances: 8550
Instances: 8600
Instances: 8650
Instances: 8700
Instances: 8750
Instances: 8800
Instances: 8850
Training CER 0.005828899061280835
Test CER 0.12601087287187723
Epoch  24
Instances: 8900
Instances: 8950
Instances: 9000
Instances: 9050
Instances: 9100
Instances: 9150
Instances: 9200
Instances: 9250
Training CER 0.008409532449846858
Test CER 0.13011600814193255
Epoch  25
Instances: 9300
Instances: 9350
Instances: 9400
Instances: 9450
Instances: 9500
Instances: 9550
Instances: 9600
Instances: 9650
Training CER 0.0074475632927812315
Test CER 0.12441653766766805
Saving Best
Epoch  26
Instances: 9700
Instances: 9750
Instances: 9800
Instances: 9850
Instances: 9900
Instances: 9950
Instances: 10000
Training CER 0.0058727711258257514
Test CER 0.12178848040529906
Saving Best
Epoch  27
Instances: 10050
Instances: 10100
Instances: 10150
Instances: 10200
Instances: 10250
Instances: 10300
Instances: 10350
Instances: 10400
Training CER 0.004340410738636915
Test CER 0.12005558251745792
Saving Best
Epoch  28
Instances: 10450
Instances: 10500
Instances: 10550
Instances: 10600
Instances: 10650
Instances: 10700
Instances: 10750
Instances: 10800
Training CER 0.005980518269724122
Test CER 0.11992633331395171
Saving Best
Epoch  29
Instances: 10850
Instances: 10900
Instances: 10950
Instances: 11000
Instances: 11050
Instances: 11100
Instances: 11150
Training CER 0.005264788305050288
Test CER 0.1295711748696354
Epoch  30
Instances: 11200
Instances: 11250
Instances: 11300
Instances: 11350
Instances: 11400
Instances: 11450
Instances: 11500
Instances: 11550
Training CER 0.005950988774996298
Test CER 0.11836344476438197
Saving Best
Saving Best
Epoch  31
Instances: 11600
Instances: 11650
Instances: 11700
Instances: 11750
Instances: 11800
Instances: 11850
Instances: 11900
Instances: 11950
Training CER 0.004723637496953204
Test CER 0.12129286601355604
Epoch  32
Instances: 12000
Instances: 12050
Instances: 12100
Instances: 12150
Instances: 12200
Instances: 12250
Instances: 12300
Instances: 12350
Training CER 0.006318954987365448
Test CER 0.1238988199931651
Epoch  33
Instances: 12400
Instances: 12450
Instances: 12500
Instances: 12550
Instances: 12600
Instances: 12650
Instances: 12700
Training CER 0.0048866378342951634
Test CER 0.11948054801698489
Epoch  34
Instances: 12750
Instances: 12800
Instances: 12850
Instances: 12900
Instances: 12950
Instances: 13000
Instances: 13050
Instances: 13100
Training CER 0.0024824235080423253
Test CER 0.11724173429092336
Saving Best
Epoch  35
Instances: 13150
Instances: 13200
Instances: 13250
Instances: 13300
Instances: 13350
Instances: 13400
Instances: 13450
Instances: 13500
Training CER 0.004849688781538229
Test CER 0.12230563295848644
Saving Best
Epoch  36
Instances: 13550
Instances: 13600
Instances: 13650
Instances: 13700
Instances: 13750
Instances: 13800
Instances: 13850
Training CER 0.00580183244853155
Test CER 0.11313459668956523
Saving Best
Epoch  37
Instances: 13900
Instances: 13950
Instances: 14000
Instances: 14050
Instances: 14100
Instances: 14150
Instances: 14200
Instances: 14250
Training CER 0.002762039730072225
Test CER 0.11427785417631219
Epoch  38
Instances: 14300
Instances: 14350
Instances: 14400
Instances: 14450
Instances: 14500
Instances: 14550
Instances: 14600
Instances: 14650
Training CER 0.0032269043447333266
Test CER 0.16907411840872946
Epoch  39
Instances: 14700
Instances: 14750
Instances: 14800
Instances: 14850
Instances: 14900
Instances: 14950
Instances: 15000
Instances: 15050
Training CER 0.007251731130767845
Test CER 0.11500763306984539
Epoch  40
Instances: 15100
Instances: 15150
Instances: 15200
Instances: 15250
Instances: 15300
Instances: 15350
Instances: 15400
Training CER 0.0029302068082563207
Test CER 0.1104741255950588
Saving Best
Saving Best
Epoch  41
Instances: 15450
Instances: 15500
Instances: 15550
Instances: 15600
Instances: 15650
Instances: 15700
Instances: 15750
Instances: 15800
Training CER 0.0020229205775660133
Test CER 0.11631009429781525
Epoch  42
Instances: 15850
Instances: 15900
Instances: 15950
Instances: 16000
Instances: 16050
Instances: 16100
Instances: 16150
Instances: 16200
Training CER 0.0033306627162110027
Test CER 0.12574476687777783
Epoch  43
Instances: 16250
Instances: 16300
Instances: 16350
Instances: 16400
Instances: 16450
Instances: 16500
Instances: 16550
Training CER 0.004793408346871549
Test CER 0.114894041304813
Epoch  44
Instances: 16600
Instances: 16650
Instances: 16700
Instances: 16750
Instances: 16800
Instances: 16850
Instances: 16900
Instances: 16950
Training CER 0.003666501947794347
Test CER 0.11640893063952067
Epoch  45
Instances: 17000
Instances: 17050
Instances: 17100
Instances: 17150
Instances: 17200
Instances: 17250
Instances: 17300
Instances: 17350
Training CER 0.0027097124359478277
Test CER 0.10969248429296388
Saving Best
Saving Best
Epoch  46
Instances: 17400
Instances: 17450
Instances: 17500
Instances: 17550
Instances: 17600
Instances: 17650
Instances: 17700
Instances: 17750
Training CER 0.0028089339023482794
Test CER 0.12057751663039484
Epoch  47
Instances: 17800
Instances: 17850
Instances: 17900
Instances: 17950
Instances: 18000
Instances: 18050
Instances: 18100
Training CER 0.007537237013206082
Test CER 0.11725139005472084
Epoch  48
Instances: 18150
Instances: 18200
Instances: 18250
Instances: 18300
Instances: 18350
Instances: 18400
Instances: 18450
Instances: 18500
Training CER 0.002432270935217821
Test CER 0.10815969941716157
Saving Best
Epoch  49
Instances: 18550
Instances: 18600
Instances: 18650
Instances: 18700
Instances: 18750
Instances: 18800
Instances: 18850
Instances: 18900
Training CER 0.0008742666911200443
Test CER 0.10974138841288489
Epoch  50
Instances: 18950
Instances: 19000
Instances: 19050
Instances: 19100
Instances: 19150
Instances: 19200
Instances: 19250
Instances: 19300
Training CER 0.0009078361408780655
Test CER 0.10916449829626829
Saving Best
Epoch  51
Instances: 19350
Instances: 19400
Instances: 19450
Instances: 19500
Instances: 19550
Instances: 19600
Instances: 19650
Training CER 0.005574972767734506
Test CER 0.11694000703146197
Epoch  52
Instances: 19700
Instances: 19750
Instances: 19800
Instances: 19850
Instances: 19900
Instances: 19950
Instances: 20000
Instances: 20050
Training CER 0.0033392337779841004
Test CER 0.11131524296604328
Epoch  53
Instances: 20100
Instances: 20150
Instances: 20200
Instances: 20250
Instances: 20300
Instances: 20350
Instances: 20400
Instances: 20450
Training CER 0.0015938577259785385
Test CER 0.11000391270033776
Epoch  54
Instances: 20500
Instances: 20550
Instances: 20600
Instances: 20650
Instances: 20700
Instances: 20750
Instances: 20800
Training CER 0.0019205340000276719
Test CER 0.11472049204568527
Epoch  55
Instances: 20850
Instances: 20900
Instances: 20950
Instances: 21000
Instances: 21050
Instances: 21100
Instances: 21150
Instances: 21200
Training CER 0.00234533024941791
Test CER 0.10954468567638376
Saving Best
Epoch  56
Instances: 21250
Instances: 21300
Instances: 21350
Instances: 21400
Instances: 21450
Instances: 21500
Instances: 21550
Instances: 21600
Training CER 0.003511369290790769
Test CER 0.11511024782695592
Epoch  57
Instances: 21650
Instances: 21700
Instances: 21750
Instances: 21800
Instances: 21850
Instances: 21900
Instances: 21950
Instances: 22000
Training CER 0.0031447845935361667
Test CER 0.11110050143543049
Epoch  58
Instances: 22050
Instances: 22100
Instances: 22150
Instances: 22200
Instances: 22250
Instances: 22300
Instances: 22350
Training CER 0.0015244823832821163
Test CER 0.10682640055196803
Saving Best
Epoch  59
Instances: 22400
Instances: 22450
Instances: 22500
Instances: 22550
Instances: 22600
Instances: 22650
Instances: 22700
Instances: 22750
Training CER 0.0017903051373094397
Test CER 0.11099840072617129
Epoch  60
Instances: 22800
Instances: 22850
Instances: 22900
Instances: 22950
Instances: 23000
Instances: 23050
Instances: 23100
Instances: 23150
Training CER 0.003412769723223945
Test CER 0.10785757231433843
Saving Best
Epoch  61
Instances: 23200
Instances: 23250
Instances: 23300
Instances: 23350
Instances: 23400
Instances: 23450
Instances: 23500
Training CER 0.0030902379040725846
Test CER 0.10711108153126714
Epoch  62
Instances: 23550
Instances: 23600
Instances: 23650
Instances: 23700
Instances: 23750
Instances: 23800
Instances: 23850
Instances: 23900
Training CER 0.0016096225889425638
Test CER 0.10738898859874883
Epoch  63
Instances: 23950
Instances: 24000
Instances: 24050
Instances: 24100
Instances: 24150
Instances: 24200
Instances: 24250
Instances: 24300
Training CER 0.0011510125617424846
Test CER 0.10845090543511715
Epoch  64
Instances: 24350
Instances: 24400
Instances: 24450
Instances: 24500
Instances: 24550
Instances: 24600
Instances: 24650
Instances: 24700
Training CER 0.0022655428827614
Test CER 0.1191339041590979
Epoch  65
Instances: 24750
Instances: 24800
Instances: 24850
Instances: 24900
Instances: 24950
Instances: 25000
Instances: 25050
Training CER 0.0031652959000786655
Test CER 0.1079300598687148
Saving Best
Epoch  66
Instances: 25100
Instances: 25150
Instances: 25200
Instances: 25250
Instances: 25300
Instances: 25350
Instances: 25400
Instances: 25450
Training CER 0.0020846184051069726
Test CER 0.1055285770531858
Saving Best
Epoch  67
Instances: 25500
Instances: 25550
Instances: 25600
Instances: 25650
Instances: 25700
Instances: 25750
Instances: 25800
Instances: 25850
Training CER 0.002017773703808005
Test CER 0.1098254435009195
Epoch  68
Instances: 25900
Instances: 25950
Instances: 26000
Instances: 26050
Instances: 26100
Instances: 26150
Instances: 26200
Training CER 0.002047668681299751
Test CER 0.1056513622625802
Epoch  69
Instances: 26250
Instances: 26300
Instances: 26350
Instances: 26400
Instances: 26450
Instances: 26500
Instances: 26550
Instances: 26600
Training CER 0.0018849862731532964
Test CER 0.11147270656404144
Epoch  70
Instances: 26650
Instances: 26700
Instances: 26750
Instances: 26800
Instances: 26850
Instances: 26900
Instances: 26950
Instances: 27000
Training CER 0.0026906116627243653
Test CER 0.10672649103625517
Saving Best
Epoch  71
Instances: 27050
Instances: 27100
Instances: 27150
Instances: 27200
Instances: 27250
Instances: 27300
Instances: 27350
Instances: 27400
Training CER 0.00210217994859592
Test CER 0.10438626974782407
Saving Best
Epoch  72
Instances: 27450
Instances: 27500
Instances: 27550
Instances: 27600
Instances: 27650
Instances: 27700
Instances: 27750
Training CER 0.0010608064513298735
Test CER 0.10227581285523987
Saving Best
Epoch  73
Instances: 27800
Instances: 27850
Instances: 27900
Instances: 27950
Instances: 28000
Instances: 28050
Instances: 28100
Instances: 28150
Training CER 0.0008129037218462018
Test CER 0.10531169860528791
Epoch  74
Instances: 28200
Instances: 28250
Instances: 28300
Instances: 28350
Instances: 28400
Instances: 28450
Instances: 28500
Instances: 28550
Training CER 0.0033665498226325597
Test CER 0.111146343558868
Epoch  75
Instances: 28600
Instances: 28650
Instances: 28700
Instances: 28750
Instances: 28800
Instances: 28850
Instances: 28900
Instances: 28950
Training CER 0.002597779716620107
Test CER 0.10644563858223266
Saving Best
Epoch  76
Instances: 29000
Instances: 29050
Instances: 29100
Instances: 29150
Instances: 29200
Instances: 29250
Instances: 29300
Training CER 0.0013504431883416164
Test CER 0.1093094169489456
Epoch  77
Instances: 29350
Instances: 29400
Instances: 29450
Instances: 29500
Instances: 29550
Instances: 29600
Instances: 29650
Instances: 29700
Training CER 0.000930792102227879
Test CER 0.10425136436434528
Epoch  78
Instances: 29750
Instances: 29800
Instances: 29850
Instances: 29900
Instances: 29950
Instances: 30000
Instances: 30050
Instances: 30100
Training CER 0.0012933158018721864
Test CER 0.11211016307188056
Epoch  79
Instances: 30150
Instances: 30200
Instances: 30250
Instances: 30300
Instances: 30350
Instances: 30400
Instances: 30450
Training CER 0.004411316703301125
Test CER 0.11106997163327775
Epoch  80
Instances: 30500
Instances: 30550
Instances: 30600
Instances: 30650
Instances: 30700
Instances: 30750
Instances: 30800
Instances: 30850
Training CER 0.001945937616496667
Test CER 0.10679820411211972
Saving Best
Epoch  81
Instances: 30900
Instances: 30950
Instances: 31000
Instances: 31050
Instances: 31100
Instances: 31150
Instances: 31200
Instances: 31250
Training CER 0.0008031523099408969
Test CER 0.10119218061689611
Saving Best
Epoch  82
Instances: 31300
Instances: 31350
Instances: 31400
Instances: 31450
Instances: 31500
Instances: 31550
Instances: 31600
Instances: 31650
Training CER 0.0005560218316750379
Test CER 0.10263510857025428
Epoch  83
Instances: 31700
Instances: 31750
Instances: 31800
Instances: 31850
Instances: 31900
Instances: 31950
Instances: 32000
Training CER 0.0002698185819172627
Test CER 0.09976251272613619
Saving Best
Epoch  84
Instances: 32050
Instances: 32100
Instances: 32150
Instances: 32200
Instances: 32250
Instances: 32300
Instances: 32350
Instances: 32400
Training CER 0.0022259631126188774
Test CER 0.1126633043669575
Epoch  85
Instances: 32450
Instances: 32500
Instances: 32550
Instances: 32600
Instances: 32650
Instances: 32700
Instances: 32750
Instances: 32800
Training CER 0.004300965602461347
Test CER 0.10694101067007554
Saving Best
Epoch  86
Instances: 32850
Instances: 32900
Instances: 32950
Instances: 33000
Instances: 33050
Instances: 33100
Instances: 33150
Training CER 0.0008536739435620188
Test CER 0.10076958351464617
Epoch  87
Instances: 33200
Instances: 33250
Instances: 33300
Instances: 33350
Instances: 33400
Instances: 33450
Instances: 33500
Instances: 33550
Training CER 0.0002286434304109525
Test CER 0.10299488145425106
Epoch  88
Instances: 33600
Instances: 33650
Instances: 33700
Instances: 33750
Instances: 33800
Instances: 33850
Instances: 33900
Instances: 33950
Training CER 0.0006231992203019514
Test CER 0.09936701144547867
Saving Best
Epoch  89
Instances: 34000
Instances: 34050
Instances: 34100
Instances: 34150
Instances: 34200
Instances: 34250
Instances: 34300
Instances: 34350
Training CER 0.002607937831607292
Test CER 0.1083798443269865
Epoch  90
Instances: 34400
Instances: 34450
Instances: 34500
Instances: 34550
Instances: 34600
Instances: 34650
Instances: 34700
Training CER 0.002583608702937389
Test CER 0.10230243299183676
Saving Best
Epoch  91
Instances: 34750
Instances: 34800
Instances: 34850
Instances: 34900
Instances: 34950
Instances: 35000
Instances: 35050
Instances: 35100
Training CER 0.0015898382079571987
Test CER 0.10289122783960003
Epoch  92
Instances: 35150
Instances: 35200
Instances: 35250
Instances: 35300
Instances: 35350
Instances: 35400
Instances: 35450
Instances: 35500
Training CER 0.0009352380721177012
Test CER 0.10440833105467931
Epoch  93
Instances: 35550
Instances: 35600
Instances: 35650
Instances: 35700
Instances: 35750
Instances: 35800
Instances: 35850
Training CER 0.0009685014333150599
Test CER 0.10585269430553715
Epoch  94
Instances: 35900
Instances: 35950
Instances: 36000
Instances: 36050
Instances: 36100
Instances: 36150
Instances: 36200
Instances: 36250
Training CER 0.0010935190576117906
Test CER 0.10403600979874153
Epoch  95
Instances: 36300
Instances: 36350
Instances: 36400
Instances: 36450
Instances: 36500
Instances: 36550
Instances: 36600
Instances: 36650
Training CER 0.0018520396642703362
Test CER 0.10556387190945908
Saving Best
Epoch  96
Instances: 36700
Instances: 36750
Instances: 36800
Instances: 36850
Instances: 36900
Instances: 36950
Instances: 37000
Instances: 37050
Training CER 0.0022981596866759727
Test CER 0.10709756206189298
Epoch  97
Instances: 37100
Instances: 37150
Instances: 37200
Instances: 37250
Instances: 37300
Instances: 37350
Instances: 37400
Training CER 0.0012107861202955
Test CER 0.10282794902331702
Epoch  98
Instances: 37450
Instances: 37500
Instances: 37550
Instances: 37600
Instances: 37650
Instances: 37700
Instances: 37750
Instances: 37800
Training CER 0.0007988922744395304
Test CER 0.09884838614282546
Saving Best
Epoch  99
Instances: 37850
Instances: 37900
Instances: 37950
Instances: 38000
Instances: 38050
Instances: 38100
Instances: 38150
Instances: 38200
Training CER 0.0005537417076421907
Test CER 0.10023971157670392
Epoch  100
Instances: 38250
Instances: 38300
Instances: 38350
Instances: 38400
Instances: 38450
Instances: 38500
Instances: 38550
Instances: 38600
Training CER 0.001406804281379698
Test CER 0.1078757627278297
Saving Best
Epoch  101
Instances: 38650
Instances: 38700
Instances: 38750
Instances: 38800
Instances: 38850
Instances: 38900
Instances: 38950
Training CER 0.002592884775215583
Test CER 0.10202048315373369
Epoch  102
Instances: 39000
Instances: 39050
Instances: 39100
Instances: 39150
Instances: 39200
Instances: 39250
Instances: 39300
Instances: 39350
Training CER 0.001047913686728055
Test CER 0.10276565377019087
Epoch  103
Instances: 39400
Instances: 39450
Instances: 39500
Instances: 39550
Instances: 39600
Instances: 39650
Instances: 39700
Instances: 39750
Training CER 0.0008262985576446068
Test CER 0.09879818740501195
Saving Best
Epoch  104
Instances: 39800
Instances: 39850
Instances: 39900
Instances: 39950
Instances: 40000
Instances: 40050
Instances: 40100
Training CER 0.0018551400434726401
Test CER 0.1389672681245999
Epoch  105
Instances: 40150
Instances: 40200
Instances: 40250
Instances: 40300
Instances: 40350
Instances: 40400
Instances: 40450
Instances: 40500
Training CER 0.002544544673837283
Test CER 0.09993486625122708
Saving Best
Epoch  106
Instances: 40550
Instances: 40600
Instances: 40650
Instances: 40700
Instances: 40750
Instances: 40800
Instances: 40850
Instances: 40900
Training CER 0.0003983475881301203
Test CER 0.09826159106433216
Saving Best
Epoch  107
Instances: 40950
Instances: 41000
Instances: 41050
Instances: 41100
Instances: 41150
Instances: 41200
Instances: 41250
Instances: 41300
Training CER 0.00012821154223309
Test CER 0.09555388393425787
Saving Best
Epoch  108
Instances: 41350
Instances: 41400
Instances: 41450
Instances: 41500
Instances: 41550
Instances: 41600
Instances: 41650
Training CER 6.282975469923864e-05
Test CER 0.09332704496674347
Saving Best
Epoch  109
Instances: 41700
Instances: 41750
Instances: 41800
Instances: 41850
Instances: 41900
Instances: 41950
Instances: 42000
Instances: 42050
Training CER 0.0028568653368120467
Test CER 0.10810075864599958
Epoch  110
Instances: 42100
Instances: 42150
Instances: 42200
Instances: 42250
Instances: 42300
Instances: 42350
Instances: 42400
Instances: 42450
Training CER 0.0020274104937076593
Test CER 0.10184429799052258
Saving Best
Epoch  111
Instances: 42500
Instances: 42550
Instances: 42600
Instances: 42650
Instances: 42700
Instances: 42750
Instances: 42800
Training CER 0.0006586895198299499
Test CER 0.09976086277747614
Epoch  112
Instances: 42850
Instances: 42900
Instances: 42950
Instances: 43000
Instances: 43050
Instances: 43100
Instances: 43150
Instances: 43200
Training CER 0.0006099008601887149
Test CER 0.09824549584053338
Epoch  113
Instances: 43250
Instances: 43300
Instances: 43350
Instances: 43400
Instances: 43450
Instances: 43500
Instances: 43550
Instances: 43600
Training CER 0.0006254645919281046
Test CER 0.09901007064846476
Epoch  114
Instances: 43650
Instances: 43700
Instances: 43750
Instances: 43800
Instances: 43850
Instances: 43900
Instances: 43950
Instances: 44000
Training CER 0.001049826634232198
Test CER 0.10369317742576524
Epoch  115
Instances: 44050
Instances: 44100
Instances: 44150
Instances: 44200
Instances: 44250
Instances: 44300
Instances: 44350
Training CER 0.002129981072716249
Test CER 0.10267942210187994
Saving Best
Epoch  116
Instances: 44400
Instances: 44450
Instances: 44500
Instances: 44550
Instances: 44600
Instances: 44650
Instances: 44700
Instances: 44750
Training CER 0.0013534610804315289
Test CER 0.10032327341995344
Epoch  117
Instances: 44800
Instances: 44850
Instances: 44900
Instances: 44950
Instances: 45000
Instances: 45050
Instances: 45100
Instances: 45150
Training CER 0.0006440842686610592
Test CER 0.09966664371255782
Epoch  118
Instances: 45200
Instances: 45250
Instances: 45300
Instances: 45350
Instances: 45400
Instances: 45450
Instances: 45500
Training CER 0.00023786759994230496
Test CER 0.09465889922170811
Epoch  119
Instances: 45550
Instances: 45600
Instances: 45650
Instances: 45700
Instances: 45750
Instances: 45800
Instances: 45850
Instances: 45900
Training CER 6.542026259065268e-05
Test CER 0.09339669212016669
Epoch  120
Instances: 45950
Instances: 46000
Instances: 46050
Instances: 46100
Instances: 46150
Instances: 46200
Instances: 46250
Instances: 46300
Training CER 3.728492001384048e-05
Test CER 0.09376166019343814
Saving Best
Epoch  121
Instances: 46350
Instances: 46400
Instances: 46450
Instances: 46500
Instances: 46550
Instances: 46600
Instances: 46650
Instances: 46700
Training CER 3.7746816999656504e-06
Test CER 0.09337556442379873
Epoch  122
Instances: 46750
Instances: 46800
Instances: 46850
Instances: 46900
Instances: 46950
Instances: 47000
Instances: 47050
Training CER 0.0014011929988542214
Test CER 0.13590747509441461
Epoch  123
Instances: 47100
Instances: 47150
Instances: 47200
Instances: 47250
Instances: 47300
Instances: 47350
Instances: 47400
Instances: 47450
Training CER 0.005910783880458169
Test CER 0.1010318514982306
Epoch  124
Instances: 47500
Instances: 47550
Instances: 47600
Instances: 47650
Instances: 47700
Instances: 47750
Instances: 47800
Instances: 47850
Training CER 0.0008090158212028365
Test CER 0.09609456368616234
Epoch  125
Instances: 47900
Instances: 47950
Instances: 48000
Instances: 48050
Instances: 48100
Instances: 48150
Instances: 48200
Instances: 48250
Training CER 0.00039878233051704904
Test CER 0.09767542675206334
Saving Best
Epoch  126
Instances: 48300
Instances: 48350
Instances: 48400
Instances: 48450
Instances: 48500
slurmstepd: error: *** JOB 31423065 ON m9g-1-1 CANCELLED AT 2019-07-13T21:45:40 ***
