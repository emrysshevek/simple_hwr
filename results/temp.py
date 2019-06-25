import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import json

n_original = 6161
n_augmented = 11001
n_online = 3388

with open('baseline/losses.json') as fp:
    baseline = json.load(fp)['test']

with open('warp/losses.json') as fp:
    warp = json.load(fp)['test']

with open('augmented_train/losses.json') as fp:
    augmented = json.load(fp)['test']

with open('augmented_train_warp/losses.json') as fp:
    augmented_warp = json.load(fp)['test']

with open('all_online/losses.json') as fp:
    all_online = json.load(fp)['test']

plt.figure()
plt.plot([i*n_original for i in range(1, len(baseline)+1)], baseline, label='baseline')
plt.plot([i*n_original for i in range(1, len(warp)+1)], warp, label='warp')
plt.plot([i*n_augmented for i in range(1, len(augmented)+1)], augmented, label='online_augmented')
plt.plot([i*n_augmented for i in range(1, len(augmented_warp)+1)], augmented_warp, label='online_augmented+warp')
plt.plot([i*n_online for i in range(1, len(all_online)+1)], all_online, label='all_online')
plt.legend()
plt.ylabel("CER")
plt.xlabel("Number of Instances")
plt.title("Loss Comparison")
plt.ylim(0, .2)
plt.xlim(0, 3000000)
plt.savefig('loss_comparison.png')
plt.close()