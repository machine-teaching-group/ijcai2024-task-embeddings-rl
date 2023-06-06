from nps.data_grouped import load_input_file, get_minibatch, shuffle_dataset
import numpy as np
vocabulary_path = "/AIML/misc/work/gtzannet/misc/GandRL_for_NPS/data/1m_6ex_karel/new_vocab.vocab"
dataset_path = "/AIML/misc/work/gtzannet/misc/GandRL_for_NPS/data/1m_6ex_karel/val.json"
batch_size = 8

dataset, vocab = load_input_file(dataset_path, vocabulary_path)
tgt_start = vocab["tkn2idx"]["<s>"]
tgt_end = vocab["tkn2idx"]["m)"]
tgt_pad = vocab["tkn2idx"]["<pad>"]

in_grouped = []
out_grouped = []
pairs = shuffle_dataset(dataset, batch_size, randomize=False)

var_lengths = set()
for i in range(len(pairs[1])):
    var_lengths.add(len(pairs[:][1][i]))
print(var_lengths)
# pairs_np = np.array(pairs)
for i in range(len(pairs[1])):
    if len(pairs[:][1][i]) == 74:
        out_grouped.append(pairs[:][0][i])
        in_grouped.append(pairs[:][1][i])
print(len(out_grouped))
print(len(in_grouped))
# grouped = pairs[len(pairs[1]) == 6]
# print(grouped)
#    print(len(pairs[1][i]))
# grouped = [pairs if pairs[1][i]==6 i in range(len(pairs[0]))]

# grouped = filter(lambda c: len(c[1]) == 6, pairs)
# print(len(list(grouped)))


# partof_dataset = dict()
# for (key, value) in dataset.items():
#    if key == "targets" and len(value) == 6:
#        print(value)
#        partof_dataset[key] = value

# print(partof_dataset)


# for i in range(len(dataset["targets"])):
#    if len(dataset["targets"][i]) == 6:
#        group_dataset_sources = dataset["sources"][i]

# for i in range(len(dataset)):
# print([len(value) for value in dataset.values()][5])
