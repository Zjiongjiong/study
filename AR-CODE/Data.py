import json
import os

import numpy
import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data import SubsetRandomSampler, DataLoader


#  Data
src_vocab_size = 1004
tgt_vocab_size = 1004
src_len = 50  # enc_input max sequence length
tgt_len = 51

angle_name='CA_C_N_angle'


def generate_test_data(angle_name):
    source_list = []
    target_list = []
    target_path = '../data/CASP12/'
    target_name = os.listdir(target_path)
    l = [x for x in range(40)]
    count=0
    for target in target_name:
        with open(target_path + target, 'r') as f:
            dict_data = json.load(f)
            i = 0
            for i in range(0, 10):
                title = list(dict_data.keys())[i]
                target_title = list(dict_data.keys())[len(dict_data) - 1]
                source_data = dict_data[title]['Angles'][angle_name]
                target_data = dict_data[target_title]['Angles'][angle_name]

                for j in range(0, len(source_data)):
                    source_list.append(source_data[j])
                    target_list.append(target_data[j])
            l[count] = (target, len(source_data))
            count += 1
        f.close()
    d = dict(l)
    source_array = np.asarray(source_list).astype(float)
    target_array = np.asarray(target_list).astype(float)

    return source_array,target_array,len(source_array),d

'''
def generate_test_data(angle_name):
    source_list = []
    target_list = []
    target_path = '../data/CASP12/'
    target_name = os.listdir(target_path)
    for target in target_name:
        # print(target)
        with open(target_path + target, 'r') as f:
            dict_data = json.load(f)
            i = 0
            for i in range(0, 10):
                title = list(dict_data.keys())[i]
                target_title = list(dict_data.keys())[len(dict_data) - 1]
                source_data = dict_data[title]['Angles'][angle_name]
                target_data = dict_data[target_title]['Angles'][angle_name]
                for j in range(0, len(source_data)):
                    source_list.append(source_data[j])
                    target_list.append(target_data[j])
        f.close()
    source_array = np.asarray(source_list).astype(float)
    target_array = np.asarray(target_list).astype(float)

    return source_array,target_array,len(source_array)
'''

def generate_data(anglename):
    source_list = []
    target_list = []

    target_path = '../data/CASP12/'
    target_name = os.listdir(target_path)
    for target in target_name:
        with open(target_path + target, 'r') as f:
            dict_data = json.load(f)
            i = 0
            for i in range(10, len(dict_data) - 1):
                title = list(dict_data.keys())[i]
                target_title = list(dict_data.keys())[len(dict_data) - 1]
                source_data = dict_data[title]['Angles'][anglename]
                target_data = dict_data[target_title]['Angles'][anglename]
                for j in range(0, len(source_data)):
                    source_list.append(source_data[j])
                    target_list.append(target_data[j])
        f.close()
    source_array = np.asarray(source_list).astype(float)
    target_array = np.asarray(target_list).astype(float)
    return source_array, target_array


def get_max_min(anglename):
    l = []
    target_path = '../data/CASP12/'
    target_name = os.listdir(target_path)
    for target in target_name:
        with open(target_path + target, 'r') as f:
            dict_data = json.load(f)
            i = 0
            for i in range(0, len(dict_data)):
                title = list(dict_data.keys())[i]
                data = dict_data[title]['Angles'][anglename]
                for j in range(0, len(data)):
                    l.append(data[j])
        f.close()
    a = np.asarray(l).astype(float)

    return max(a), min(a)


def data_convertion(data,maxAngle,minAngle):
    normlist = []
    for val in data:
        normVal = (val - minAngle) / (maxAngle - minAngle)
        normlist.append(round(normVal, 3))

    normlist = [i * 1000 for i in normlist]
    normlist = numpy.asarray(normlist).astype(int)
    return normlist


def data_split(data, seqlen):
    databatch = []
    list = []
    for i in range(len(data)):
        if ((i + 1) % seqlen):
            list.append(data[i])
        else:
            list.append(data[i])
            databatch.append(list)
            list = []
        i += 1
    # databatch.append(list)
    return databatch


def make_decoderinput(data):
    for i in range(len(data)):
        data[i].insert(0, 1001)
        # data[i].append(36002)
    return data


def make_decoderoutput(data):
    for i in range(len(data)):
        # data[i].insert(0, 36001)
        data[i].append(1002)
    return data


def convert_to_Longtensor(data):
    data = torch.LongTensor(data)
    return data


def data_from_file(seqlen,angle_name):
    source, target = generate_data(angle_name)
    test_source,test_target,testlen,lendict=generate_test_data(angle_name)
    max,min=get_max_min(angle_name)
    enc_inputs = convert_to_Longtensor(data_split(data_convertion(source,max,min), seqlen))
    dec_inputs = convert_to_Longtensor(make_decoderinput(data_split(data_convertion(target,max,min), seqlen)))
    dec_outputs = convert_to_Longtensor(make_decoderoutput(data_split(data_convertion(target,max,min), seqlen)))
    test_enc_inputs = convert_to_Longtensor(data_split(data_convertion(test_source, max, min), seqlen))
    test_dec_inputs = convert_to_Longtensor(make_decoderinput(data_split(data_convertion(test_target, max, min), seqlen)))
    test_dec_outputs = convert_to_Longtensor(make_decoderoutput(data_split(data_convertion(test_target, max, min), seqlen)))

    rest=[]
    if ((len(test_enc_inputs) * seqlen) != testlen):
        d = data_convertion(test_source,max,min)
        print("rest begin```")
        print(len(test_enc_inputs))
        print(testlen)
        print(d)
        print(len(d))
        rest = d[len(test_enc_inputs) * seqlen:]

    return enc_inputs, dec_inputs, dec_outputs, test_enc_inputs,test_dec_inputs,test_dec_outputs,max, min,rest,lendict



seqlen = 50

enc_inputs, dec_inputs, dec_outputs, test_enc_inputs,test_dec_inputs,test_dec_outputs,max, min,rest,lendict = data_from_file(seqlen,angle_name)
print(len(enc_inputs))
print(len(dec_outputs))
print(len(dec_outputs))
print(enc_inputs)
print(dec_inputs)
print(dec_outputs)
# print(len(test_enc_inputs))
# print(len(test_dec_outputs))
# print(max)
# print(min)
# print("rest：", rest)
# print("lendict:\n")
# print(lendict)
# print(len(lendict))

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
loader = DataLoader(dataset, 50, False)
train_sampler, valid_sampler = dataset_split(len(dataset), random_seed=9)
train_loader = DataLoader(dataset, 50, shuffle=False, sampler=train_sampler)
valid_loader = DataLoader(dataset, 50, shuffle=False, sampler=valid_sampler)

test_dataset = MyDataSet(test_enc_inputs, test_dec_inputs, test_dec_outputs)
test_loader = DataLoader(test_dataset, 1, False)
'''

print(angle_name)
# print(len(loader))
print(len(train_loader))
print(len(valid_loader))
print(len(test_loader))

'''



