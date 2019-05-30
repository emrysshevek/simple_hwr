import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import json

n_original = 6161
n_online = 11001

with open('baseline_iam/losses.json') as fp:
    baseline = json.load(fp)['test']

with open('warp_iam/losses.json') as fp:
    warp = json.load(fp)['test']

with open('online_iam/losses.json') as fp:
    online = json.load(fp)['test']

with open('online_warp_iam/losses.json') as fp:
    online_warp = json.load(fp)['test']

plt.figure()
plt.plot([i*n_original for i in range(1, len(baseline)+1)], baseline, label='baseline')
plt.plot([i*n_original for i in range(1, len(warp)+1)], warp, label='warp')
plt.plot([i*n_online for i in range(1, len(online)+1)], online, label='online')
plt.plot([i*n_online for i in range(1, len(online_warp)+1)], online_warp, label='online+warp')
plt.legend()
plt.ylabel("CER")
plt.xlabel("Number of Instances")
plt.title("Loss Comparison")
plt.savefig('loss_comparison.png')
plt.close()