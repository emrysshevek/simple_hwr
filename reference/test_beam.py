import torch
from ctcdecode import CTCBeamDecoder
import numpy as np

labels = ["|", "c", "c"]
my_decoder = CTCBeamDecoder(labels=labels, blank_id=0, beam_width=5, num_processes=1, log_probs_input=False)
softmax = torch.nn.Softmax(dim=2)

output = np.array([[5,10,5],[100,5,5],[5,100,5],[5,100,5],[5,100,5]]) # batch x seq x label_size; each row is the label probabilities; columns are sequence length
output = softmax(torch.tensor(output[None, :,:]).float())
print(output)

def beam(out):
    print(out.shape)
    pred, scores, timesteps, out_seq_len = my_decoder.decode(out)
    print(f"output {pred}")
    print(f"scores {scores}")
    print(f"timesteps {timesteps}")
    print(f"out_seq_len {out_seq_len}")
    print(pred[0][1])
    lookup(output, out_seq_len)

def lookup(output, output_lengths, indexToCharacter):

    # Loop through batches
    for batch in range(output.size[0]):
        line = output[batch][:output_lengths[batch]]
        string = u""
        for char in line:

            else:
                break
        else:
            val = label[i]
            string += indexToCharacter[val]
    return string

beam(output)


# import kenlm
# model = kenlm.Model('lm/test.arpa')
# print(model.score('this is a sentence .', bos = True, eos = True))
# #pip install https://github.com/kpu/kenlm/archive/master.zip
