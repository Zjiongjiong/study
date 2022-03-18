import json
import os

import numpy
import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data import SubsetRandomSampler, DataLoader


def data_concat(dir):
    source_list = []
    target_list = []

    target_path = '../data/partition_data/psi_train/%s/' % dir
    target_name = os.listdir(target_path)
    dict = {}
    for target in target_name:
        with open(target_path + target, 'r') as f:
            dict_data = json.load(f)
            for k in dict_data.keys():
                if k in dict.keys():
                    for l in dict_data[k]:
                        # print(l)
                        dict[k].append(l)
                else:
                    dict[k] = dict_data[k]
        f.close()

    json_str = json.dumps(dict)
    output_file = '../data/partition_data/psi_train/%s.json' % dir
    with open(output_file, "w") as json_output:
        json_output.write(json_str)
    json_output.close()
    return True


# data_concat('C_source')
# data_concat('C_target')
# data_concat('E_source')
# data_concat('E_target')
# data_concat('H_source')
# data_concat('H_target')

def load_input_seq(ss,length):
    source_file = '../data/partition_data/psi_train/%s_source.json' % ss
    target_file = '../data/partition_data/psi_train/%s_target.json' % ss
    with open(source_file, 'r') as sf:
        with open(target_file, 'r') as tf:
            sdict_data = json.load(sf)
            tdict_data = json.load(tf)
            sl=sdict_data[length]
            tl=tdict_data[length]
        tf.close()
    sf.close()
    return sl,tl


def make_decoderinput(data):
    for i in range(len(data)):
        data[i].insert(0, 1001)
        # data[i].append(36002)
    return data


def recover_target(data):
    for i in range(len(data)):
        del data[i][0]
    return data


def make_decoderoutput(data):
    for i in range(len(data)):
        # data[i].insert(0, 36001)
        data[i].append(1002)
    return data


def convert_to_Longtensor(data):
    data = torch.LongTensor(data)
    return data


ss = 'C'
index=1
file = '../data/partition_data/psi_train/%s_source.json' % ss
with open(file, 'r') as f:
    dict_data = json.load(f)
    print(len(dict_data))
    # for k in dict_data.keys():
    #     print(k,'------------',len(dict_data[k]),end='  ')
    length = list(dict_data.keys())[index]
    print(length,'---------',len(dict_data[length]))
f.close()

sl,tl=load_input_seq(ss,length)
print(sl)
print(tl)
print(len(sl))
print(len(tl))

enc_inputs = convert_to_Longtensor(sl)
dec_inputs = convert_to_Longtensor(make_decoderinput(tl))
tl=recover_target(tl)
dec_outputs = convert_to_Longtensor(make_decoderoutput(tl))

print(enc_inputs)
print(dec_inputs)
print(dec_outputs)
print(len(enc_inputs))
print(len(dec_inputs))
print(len(dec_outputs))

class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


# 划分训练集、验证集、测试集（7：1：2）
def dataset_split(dataset_size, random_seed):
    split1 = .7
    # split2 = .8

    indices = list(range(dataset_size))
    train_split = int(np.floor(split1 * dataset_size))
    # valid_split = int(np.floor(split2 * dataset_size))

    shuffle_dataset = True
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    # train_indices, valid_indices, test_indices = indices[: train_split], indices[train_split: valid_split], indices[valid_split:]
    train_indices, valid_indices = indices[: train_split], indices[train_split:]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    # test_sampler = SubsetRandomSampler(test_indices)

    return train_sampler, valid_sampler


dataset = MyDataSet(enc_inputs, dec_inputs, dec_outputs)
loader = DataLoader(dataset, 10, False)
train_sampler, valid_sampler = dataset_split(len(dataset), random_seed=9)
train_loader = DataLoader(dataset, 10, shuffle=False, sampler=train_sampler)
valid_loader = DataLoader(dataset, 10, shuffle=False, sampler=valid_sampler)

# test_dataset = MyDataSet(test_enc_inputs, test_dec_inputs, test_dec_outputs)
# test_loader = DataLoader(test_dataset, 1, False)


# print(angle_name)
print('---DATA(SS_LENGTH)---:  '+ss+'_'+length)
# print(len(loader))
print(len(train_loader))
print(len(valid_loader))
# print(len(test_loader))

