import sys
import json
import os
import numpy as np
from collections import defaultdict
import string

PAD_TOKEN = '<pad>'
PAD_IDX = None
SOS_TOKEN = '<sos>'
SOS_IDX = None
EOS_TOKEN = '<eos>'
EOS_IDX = None


def load_char_set(char_set_path):
    with open(char_set_path) as f:
        char_set = json.load(f)

    idx_to_char = {}
    for k,v in char_set['idx_to_char'].iteritems():
        idx_to_char[int(k)] = v

    return idx_to_char, char_set['char_to_idx']


def make_universal_char_set():
    tokens = ['<PAD>'] + [c for c in string.printable][:-2] + ['<SOS>', '<EOS>']

    # move empty token to front for ctc purposes
    # tokens = [tokens.pop(tokens.index('|'))] + tokens

    idx_to_char = {i: c for i, c in enumerate(tokens)}
    char_to_idx = {c: i for i, c in enumerate(tokens)}

    return char_to_idx, idx_to_char, char_to_idx['<SOS>'], char_to_idx['<EOS>'], char_to_idx['<PAD>']


def make_char_set(paths, root="./data", seq2seq=False):
    out_char_to_idx = {}
    out_idx_to_char = {}
    char_counts = defaultdict(int)
    instance_count = 0
    for data_file in paths:
        with open(os.path.join(root, data_file)) as f:
            data = json.load(f)

        cnt = 1  # this is important that this starts at 1 not 0
        for data_item in data:
            for c in data_item.get('gt', ""):
                if c not in out_char_to_idx:
                    out_char_to_idx[c] = cnt
                    out_idx_to_char[cnt] = c
                    cnt += 1
                char_counts[c] += 1
            instance_count += 1

    out_char_to_idx2 = {}
    out_idx_to_char2 = {}

    for i, c in enumerate(sorted(out_char_to_idx.keys())):
        out_char_to_idx2[c] = i + 1
        out_idx_to_char2[i + 1] = c

    # Add empty
    out_char_to_idx2["|"] = 0
    out_idx_to_char2[0] = "|"

    # If using seq2seq model, add <pad>, <sos> and <eos> tags
    # if seq2seq:
    n_chars = len(out_char_to_idx2)
    sos_token, eos_token, pad_token = '<SOS>', '<EOS>', '<PAD>'
    sos_idx, eos_idx, pad_idx = n_chars, n_chars + 1, n_chars + 2

    out_char_to_idx2[sos_token] = sos_idx
    out_idx_to_char2[sos_idx] = sos_token

    out_char_to_idx2[eos_token] = eos_idx
    out_idx_to_char2[eos_idx] = eos_token

    out_char_to_idx2[pad_token] = pad_idx
    out_idx_to_char2[pad_idx] = pad_token

    char_counts[eos_token] = instance_count
    char_freq = np.zeros(len(out_char_to_idx2))
    total_chars = sum(count for count in char_counts.values())
    for c, count in char_counts.items():
        char_freq[out_char_to_idx2[c]] = count/total_chars
    # print(char_freq)

    return out_char_to_idx2, out_idx_to_char2, char_freq, sos_idx, eos_idx, pad_idx


if __name__ == "__main__":
    character_set_path = sys.argv[-1]
    paths = [sys.argv[i] for i in range(1, len(sys.argv)-1)]

    char_to_idx, idx_to_char, char_freq = make_char_set(*paths)

    output_data = {
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char
    }

    for k,v in sorted(char_freq.iteritems(), key=lambda x: x[1]):
        print(k, v)

    print("Size:", len(output_data['char_to_idx']))

    with open(character_set_path, 'w') as outfile:
        json.dump(output_data, outfile)
