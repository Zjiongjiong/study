import json
import os

import numpy
import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data import SubsetRandomSampler, DataLoader
# from test import ss_list


angle_name='psi_im1'

ss_list = [[['C', 28], ['H', 10], ['C', 6], ['E', 6], ['C', 19], ['H', 23], ['C', 21]], [['C', 14], ['E', 10], ['H', 9], ['E', 6], ['C', 23], ['E', 5], ['C', 11], ['E', 24], ['C', 5], ['E', 7], ['C', 7], ['E', 8], ['C', 7]], [['C', 25], ['E', 9], ['C', 6], ['H', 14], ['C', 16], ['H', 15], ['E', 11], ['H', 12], ['E', 5], ['C', 6], ['H', 13], ['C', 15], ['H', 16], ['C', 6], ['E', 9], ['H', 18], ['E', 6], ['C', 6], ['H', 5], ['C', 37], ['H', 42], ['E', 9], ['H', 11]], [['H', 36], ['C', 7], ['H', 50]], [['C', 11], ['H', 27], ['C', 23], ['H', 20], ['C', 10], ['H', 491]], [['C', 7], ['E', 7], ['C', 5], ['H', 11], ['C', 13], ['E', 15], ['H', 8], ['C', 11], ['H', 7], ['C', 32], ['E', 5], ['C', 6], ['H', 13], ['E', 6], ['H', 8], ['E', 13], ['H', 12], ['E', 39], ['H', 15], ['E', 13]], [['C', 5], ['H', 57]], [['E', 9], ['C', 16], ['E', 30], ['C', 5], ['E', 5], ['C', 7], ['E', 6], ['C', 26], ['H', 11]], [['H', 18], ['C', 32], ['H', 61], ['E', 5]], [['C', 16], ['H', 28], ['C', 15], ['H', 45]], [['H', 8], ['C', 13], ['H', 44], ['C', 10], ['H', 48]], [['C', 15], ['H', 28], ['C', 6], ['H', 16], ['C', 13], ['H', 13], ['C', 48], ['H', 27], ['C', 16], ['E', 14], ['C', 16], ['H', 22], ['C', 7], ['E', 12], ['C', 7], ['E', 6], ['C', 9], ['H', 7], ['C', 17], ['H', 20]], [['C', 34], ['E', 5], ['C', 18], ['E', 10], ['C', 5], ['E', 11], ['C', 5]], [['C', 7], ['H', 13], ['C', 10], ['H', 20], ['C', 10], ['E', 5], ['C', 11], ['H', 8], ['C', 8], ['H', 12], ['C', 44], ['E', 9], ['C', 6], ['E', 19], ['C', 5], ['H', 11], ['C', 6], ['E', 5], ['C', 5], ['H', 5], ['C', 9], ['H', 12], ['E', 7], ['C', 10], ['E', 19], ['C', 18], ['H', 23], ['E', 19], ['H', 6], ['C', 6], ['H', 13], ['C', 8], ['E', 5], ['C', 8], ['H', 10], ['C', 33], ['E', 5], ['C', 20], ['H', 12]], [['C', 14], ['E', 7], ['C', 6], ['H', 40], ['C', 15], ['H', 26], ['C', 17], ['H', 14], ['E', 7], ['C', 11], ['H', 10], ['C', 6], ['E', 5], ['C', 5], ['H', 19], ['C', 18]], [['E', 7], ['C', 6], ['E', 18], ['C', 36], ['H', 10], ['C', 7], ['E', 6], ['C', 5], ['E', 6], ['C', 8], ['E', 5], ['C', 61], ['E', 7], ['C', 7], ['E', 6], ['H', 34]], [['C', 5], ['E', 5], ['H', 19], ['E', 8], ['H', 14], ['E', 10], ['H', 19], ['C', 18], ['H', 42], ['C', 7], ['H', 31], ['C', 14], ['H', 6], ['C', 9], ['H', 12], ['C', 20]], [['E', 9], ['C', 14], ['E', 18], ['C', 5], ['E', 17], ['C', 5], ['E', 8], ['C', 6], ['E', 9], ['C', 5], ['E', 10], ['C', 6]], [['E', 9], ['C', 7], ['H', 25], ['C', 9], ['H', 48], ['C', 21], ['E', 5], ['H', 6], ['C', 24], ['E', 6], ['H', 23], ['E', 10]], [['H', 143], ['E', 7], ['C', 7], ['E', 5], ['C', 6], ['E', 7], ['C', 7], ['H', 6], ['C', 17], ['H', 14], ['E', 6], ['C', 5], ['E', 12]], [['C', 6], ['E', 16], ['C', 9], ['E', 15], ['C', 7], ['E', 8], ['C', 10], ['E', 8], ['C', 12], ['E', 17], ['C', 9], ['E', 10], ['C', 14], ['E', 12], ['C', 129], ['E', 6], ['C', 56], ['E', 10], ['C', 36], ['E', 12], ['C', 21], ['H', 11], ['C', 13]], [['H', 44], ['E', 8], ['H', 41], ['E', 8], ['C', 7], ['H', 22], ['C', 87], ['E', 14], ['C', 6], ['E', 10], ['C', 5], ['E', 10]], [['H', 15], ['C', 6], ['H', 18], ['C', 9], ['H', 5], ['C', 15], ['H', 10], ['E', 5], ['C', 6], ['H', 17], ['E', 6], ['C', 8], ['E', 16], ['C', 25]], [['C', 5], ['E', 6], ['C', 7], ['E', 7], ['C', 19], ['E', 24], ['C', 5], ['E', 10], ['C', 19]], [['C', 11], ['H', 25], ['C', 5], ['E', 6], ['C', 20], ['E', 19], ['H', 36], ['C', 31], ['H', 13], ['C', 7], ['E', 5], ['H', 8], ['C', 68], ['E', 5], ['C', 16], ['E', 25]], [['H', 348]], [['H', 311]], [['H', 408]], [['E', 6], ['C', 18], ['H', 14], ['C', 8], ['E', 19], ['C', 10], ['E', 18], ['C', 7], ['E', 7], ['C', 11], ['E', 9], ['C', 29], ['H', 7], ['C', 5], ['H', 6], ['E', 34], ['C', 36], ['E', 14], ['C', 9], ['E', 17], ['C', 7], ['E', 11], ['C', 145], ['E', 19], ['C', 39], ['E', 6], ['C', 40], ['E', 9], ['C', 23], ['H', 16]], [['E', 8], ['C', 10], ['E', 18], ['C', 5], ['E', 11], ['C', 9], ['E', 10], ['C', 27], ['E', 5], ['C', 13], ['E', 6], ['C', 17], ['E', 11], ['C', 5], ['E', 12], ['C', 14], ['E', 17], ['H', 8], ['C', 26], ['H', 7], ['C', 5], ['E', 7], ['C', 17], ['E', 12], ['C', 11], ['E', 10], ['H', 19], ['E', 5], ['C', 24]], [['E', 8], ['C', 6], ['H', 22], ['E', 5], ['C', 5], ['H', 41], ['C', 24], ['H', 16], ['E', 9], ['H', 17], ['E', 7], ['C', 6], ['H', 14], ['E', 6], ['H', 18], ['E', 8], ['H', 15], ['C', 10], ['H', 11], ['C', 5], ['H', 40], ['E', 5], ['C', 9], ['E', 17], ['C', 8], ['H', 52], ['C', 25], ['H', 12], ['E', 7], ['C', 5], ['E', 10], ['C', 5], ['E', 10], ['H', 9], ['C', 7], ['E', 6], ['C', 14], ['H', 13], ['E', 20], ['C', 5], ['E', 8]], [['E', 26], ['C', 5], ['E', 10], ['C', 47], ['E', 9], ['C', 6], ['E', 15], ['C', 5], ['E', 15]], [['C', 16], ['H', 12], ['C', 8], ['H', 7], ['C', 10], ['H', 11], ['C', 10]], [['C', 10], ['E', 6], ['C', 5], ['E', 37], ['C', 6], ['E', 24], ['C', 12], ['E', 5], ['C', 7], ['E', 6], ['C', 5], ['E', 15], ['C', 5], ['E', 5], ['C', 6], ['E', 6], ['C', 5], ['E', 14], ['C', 5], ['E', 5], ['C', 5], ['E', 8], ['C', 14], ['E', 14], ['C', 5], ['E', 6], ['C', 5], ['E', 6], ['C', 5], ['E', 6], ['C', 5], ['E', 15], ['C', 5], ['E', 6], ['C', 6], ['E', 6], ['C', 6], ['E', 16], ['C', 13]], [['C', 71], ['H', 14], ['E', 43], ['C', 10], ['E', 11], ['C', 6], ['E', 8], ['C', 6], ['E', 17], ['C', 8], ['H', 14], ['C', 10], ['E', 21], ['C', 5], ['E', 9], ['C', 11], ['H', 5], ['C', 22], ['E', 8], ['C', 5], ['E', 8], ['C', 8], ['E', 9], ['C', 12]], [['H', 15], ['C', 8], ['H', 16], ['C', 8], ['E', 5], ['C', 19], ['H', 316]], [['C', 18], ['E', 17], ['C', 5], ['E', 5], ['C', 10], ['H', 14], ['E', 5], ['H', 23], ['C', 9], ['E', 6], ['C', 8], ['E', 12], ['H', 21], ['C', 7], ['E', 13], ['H', 13], ['E', 9], ['C', 12], ['E', 7], ['C', 5], ['H', 20], ['C', 9], ['H', 20], ['C', 8], ['H', 91], ['C', 34], ['H', 20], ['C', 22], ['E', 7], ['C', 30], ['E', 10], ['H', 13], ['C', 6]], [['C', 6], ['H', 11], ['C', 88], ['H', 10], ['E', 6], ['C', 33], ['E', 11], ['C', 7], ['E', 10], ['C', 18], ['H', 8], ['C', 7], ['H', 6], ['C', 7], ['E', 5], ['C', 20]], [['H', 31], ['C', 16], ['H', 272], ['C', 15], ['H', 41]], [['C', 9], ['H', 26], ['E', 27], ['C', 18], ['E', 8], ['C', 16], ['H', 9], ['C', 40], ['H', 10], ['C', 5], ['E', 7]]]

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
        val=float(val)
        normVal = (val - minAngle) / (maxAngle - minAngle)
        normlist.append(round(normVal, 3))

    normlist = [i * 1000 for i in normlist]
    # normlist = numpy.asarray(normlist).astype(int)
    return normlist


angle_name="psi_im1"
max,min=get_max_min(angle_name)
print(max,min)
ss_list = [[['C', 28], ['H', 10], ['C', 6], ['E', 6], ['C', 19], ['H', 23], ['C', 21]], [['C', 14], ['E', 10], ['H', 9], ['E', 6], ['C', 23], ['E', 5], ['C', 11], ['E', 24], ['C', 5], ['E', 7], ['C', 7], ['E', 8], ['C', 7]], [['C', 25], ['E', 9], ['C', 6], ['H', 14], ['C', 16], ['H', 15], ['E', 11], ['H', 12], ['E', 5], ['C', 6], ['H', 13], ['C', 15], ['H', 16], ['C', 6], ['E', 9], ['H', 18], ['E', 6], ['C', 6], ['H', 5], ['C', 37], ['H', 42], ['E', 9], ['H', 11]], [['H', 36], ['C', 7], ['H', 50]], [['C', 11], ['H', 27], ['C', 23], ['H', 20], ['C', 10], ['H', 491]], [['C', 7], ['E', 7], ['C', 5], ['H', 11], ['C', 13], ['E', 15], ['H', 8], ['C', 11], ['H', 7], ['C', 32], ['E', 5], ['C', 6], ['H', 13], ['E', 6], ['H', 8], ['E', 13], ['H', 12], ['E', 39], ['H', 15], ['E', 13]], [['C', 5], ['H', 57]], [['E', 9], ['C', 16], ['E', 30], ['C', 5], ['E', 5], ['C', 7], ['E', 6], ['C', 26], ['H', 11]], [['H', 18], ['C', 32], ['H', 61], ['E', 5]], [['C', 16], ['H', 28], ['C', 15], ['H', 45]], [['H', 8], ['C', 13], ['H', 44], ['C', 10], ['H', 48]], [['C', 15], ['H', 28], ['C', 6], ['H', 16], ['C', 13], ['H', 13], ['C', 48], ['H', 27], ['C', 16], ['E', 14], ['C', 16], ['H', 22], ['C', 7], ['E', 12], ['C', 7], ['E', 6], ['C', 9], ['H', 7], ['C', 17], ['H', 20]], [['C', 34], ['E', 5], ['C', 18], ['E', 10], ['C', 5], ['E', 11], ['C', 5]], [['C', 7], ['H', 13], ['C', 10], ['H', 20], ['C', 10], ['E', 5], ['C', 11], ['H', 8], ['C', 8], ['H', 12], ['C', 44], ['E', 9], ['C', 6], ['E', 19], ['C', 5], ['H', 11], ['C', 6], ['E', 5], ['C', 5], ['H', 5], ['C', 9], ['H', 12], ['E', 7], ['C', 10], ['E', 19], ['C', 18], ['H', 23], ['E', 19], ['H', 6], ['C', 6], ['H', 13], ['C', 8], ['E', 5], ['C', 8], ['H', 10], ['C', 33], ['E', 5], ['C', 20], ['H', 12]], [['C', 14], ['E', 7], ['C', 6], ['H', 40], ['C', 15], ['H', 26], ['C', 17], ['H', 14], ['E', 7], ['C', 11], ['H', 10], ['C', 6], ['E', 5], ['C', 5], ['H', 19], ['C', 18]], [['E', 7], ['C', 6], ['E', 18], ['C', 36], ['H', 10], ['C', 7], ['E', 6], ['C', 5], ['E', 6], ['C', 8], ['E', 5], ['C', 61], ['E', 7], ['C', 7], ['E', 6], ['H', 34]], [['C', 5], ['E', 5], ['H', 19], ['E', 8], ['H', 14], ['E', 10], ['H', 19], ['C', 18], ['H', 42], ['C', 7], ['H', 31], ['C', 14], ['H', 6], ['C', 9], ['H', 12], ['C', 20]], [['E', 9], ['C', 14], ['E', 18], ['C', 5], ['E', 17], ['C', 5], ['E', 8], ['C', 6], ['E', 9], ['C', 5], ['E', 10], ['C', 6]], [['E', 9], ['C', 7], ['H', 25], ['C', 9], ['H', 48], ['C', 21], ['E', 5], ['H', 6], ['C', 24], ['E', 6], ['H', 23], ['E', 10]], [['H', 143], ['E', 7], ['C', 7], ['E', 5], ['C', 6], ['E', 7], ['C', 7], ['H', 6], ['C', 17], ['H', 14], ['E', 6], ['C', 5], ['E', 12]], [['C', 6], ['E', 16], ['C', 9], ['E', 15], ['C', 7], ['E', 8], ['C', 10], ['E', 8], ['C', 12], ['E', 17], ['C', 9], ['E', 10], ['C', 14], ['E', 12], ['C', 129], ['E', 6], ['C', 56], ['E', 10], ['C', 36], ['E', 12], ['C', 21], ['H', 11], ['C', 13]], [['H', 44], ['E', 8], ['H', 41], ['E', 8], ['C', 7], ['H', 22], ['C', 87], ['E', 14], ['C', 6], ['E', 10], ['C', 5], ['E', 10]], [['H', 15], ['C', 6], ['H', 18], ['C', 9], ['H', 5], ['C', 15], ['H', 10], ['E', 5], ['C', 6], ['H', 17], ['E', 6], ['C', 8], ['E', 16], ['C', 25]], [['C', 5], ['E', 6], ['C', 7], ['E', 7], ['C', 19], ['E', 24], ['C', 5], ['E', 10], ['C', 19]], [['C', 11], ['H', 25], ['C', 5], ['E', 6], ['C', 20], ['E', 19], ['H', 36], ['C', 31], ['H', 13], ['C', 7], ['E', 5], ['H', 8], ['C', 68], ['E', 5], ['C', 16], ['E', 25]], [['H', 348]], [['H', 311]], [['H', 408]], [['E', 6], ['C', 18], ['H', 14], ['C', 8], ['E', 19], ['C', 10], ['E', 18], ['C', 7], ['E', 7], ['C', 11], ['E', 9], ['C', 29], ['H', 7], ['C', 5], ['H', 6], ['E', 34], ['C', 36], ['E', 14], ['C', 9], ['E', 17], ['C', 7], ['E', 11], ['C', 145], ['E', 19], ['C', 39], ['E', 6], ['C', 40], ['E', 9], ['C', 23], ['H', 16]], [['E', 8], ['C', 10], ['E', 18], ['C', 5], ['E', 11], ['C', 9], ['E', 10], ['C', 27], ['E', 5], ['C', 13], ['E', 6], ['C', 17], ['E', 11], ['C', 5], ['E', 12], ['C', 14], ['E', 17], ['H', 8], ['C', 26], ['H', 7], ['C', 5], ['E', 7], ['C', 17], ['E', 12], ['C', 11], ['E', 10], ['H', 19], ['E', 5], ['C', 24]], [['E', 8], ['C', 6], ['H', 22], ['E', 5], ['C', 5], ['H', 41], ['C', 24], ['H', 16], ['E', 9], ['H', 17], ['E', 7], ['C', 6], ['H', 14], ['E', 6], ['H', 18], ['E', 8], ['H', 15], ['C', 10], ['H', 11], ['C', 5], ['H', 40], ['E', 5], ['C', 9], ['E', 17], ['C', 8], ['H', 52], ['C', 25], ['H', 12], ['E', 7], ['C', 5], ['E', 10], ['C', 5], ['E', 10], ['H', 9], ['C', 7], ['E', 6], ['C', 14], ['H', 13], ['E', 20], ['C', 5], ['E', 8]], [['E', 26], ['C', 5], ['E', 10], ['C', 47], ['E', 9], ['C', 6], ['E', 15], ['C', 5], ['E', 15]], [['C', 16], ['H', 12], ['C', 8], ['H', 7], ['C', 10], ['H', 11], ['C', 10]], [['C', 10], ['E', 6], ['C', 5], ['E', 37], ['C', 6], ['E', 24], ['C', 12], ['E', 5], ['C', 7], ['E', 6], ['C', 5], ['E', 15], ['C', 5], ['E', 5], ['C', 6], ['E', 6], ['C', 5], ['E', 14], ['C', 5], ['E', 5], ['C', 5], ['E', 8], ['C', 14], ['E', 14], ['C', 5], ['E', 6], ['C', 5], ['E', 6], ['C', 5], ['E', 6], ['C', 5], ['E', 15], ['C', 5], ['E', 6], ['C', 6], ['E', 6], ['C', 6], ['E', 16], ['C', 13]], [['C', 71], ['H', 14], ['E', 43], ['C', 10], ['E', 11], ['C', 6], ['E', 8], ['C', 6], ['E', 17], ['C', 8], ['H', 14], ['C', 10], ['E', 21], ['C', 5], ['E', 9], ['C', 11], ['H', 5], ['C', 22], ['E', 8], ['C', 5], ['E', 8], ['C', 8], ['E', 9], ['C', 12]], [['H', 15], ['C', 8], ['H', 16], ['C', 8], ['E', 5], ['C', 19], ['H', 316]], [['C', 18], ['E', 17], ['C', 5], ['E', 5], ['C', 10], ['H', 14], ['E', 5], ['H', 23], ['C', 9], ['E', 6], ['C', 8], ['E', 12], ['H', 21], ['C', 7], ['E', 13], ['H', 13], ['E', 9], ['C', 12], ['E', 7], ['C', 5], ['H', 20], ['C', 9], ['H', 20], ['C', 8], ['H', 91], ['C', 34], ['H', 20], ['C', 22], ['E', 7], ['C', 30], ['E', 10], ['H', 13], ['C', 6]], [['C', 6], ['H', 11], ['C', 88], ['H', 10], ['E', 6], ['C', 33], ['E', 11], ['C', 7], ['E', 10], ['C', 18], ['H', 8], ['C', 7], ['H', 6], ['C', 7], ['E', 5], ['C', 20]], [['H', 31], ['C', 16], ['H', 272], ['C', 15], ['H', 41]], [['C', 9], ['H', 26], ['E', 27], ['C', 18], ['E', 8], ['C', 16], ['H', 9], ['C', 40], ['H', 10], ['C', 5], ['E', 7]]]
print(len(ss_list))
target_path = '../data/CASP12/'
target_name = os.listdir(target_path)
count = 0

for target in target_name:

    print(target)
    with open(target_path + target, 'r') as f:

        C_dict = {}
        H_dict = {}
        E_dict = {}
        C_dict_target = {}
        H_dict_target = {}
        E_dict_target = {}

        print(len(ss_list[count]))
        dict_data = json.load(f)
        print(len(dict_data)-1 - 10)
        for i in range(10, len(dict_data) - 1):
            # if count==1:
            #     print(i)
            title = list(dict_data.keys())[i]
            target_title = list(dict_data.keys())[len(dict_data) - 1]
            source_data = dict_data[title]['Angles']['psi_im1']
            target_data = dict_data[target_title]['Angles']['psi_im1']
            if i == 10:
                print(source_data)
                print(target_data)
            # source_array = np.asarray(source_data).astype(float)
            # target_array = np.asarray(target_data).astype(float)
            source_data=data_convertion(source_data,max,min)
            target_data=data_convertion(target_data,max,min)
            if i == 10:
                print(source_data)
                print(target_data)

            c = 0
            for j in range(0, len(ss_list[count])):
                s = c
                e = c + ss_list[count][j][1]
                # if i==178 and count==1:
                #     print('\n',s,e)
                #     print(ss_list[count][j][1],source_data[s:e],target_data[s:e])

                if ss_list[count][j][0]=='C':
                    if str(ss_list[count][j][1]) in C_dict.keys():
                        C_dict[str(ss_list[count][j][1])].append(source_data[s:e])
                        C_dict_target[str(ss_list[count][j][1])].append(target_data[s:e])
                    else:
                        C_dict[str(ss_list[count][j][1])] = [source_data[s:e]]
                        C_dict_target[str(ss_list[count][j][1])] = [target_data[s:e]]
                if ss_list[count][j][0]=='E':
                    if str(ss_list[count][j][1]) in E_dict.keys():
                        E_dict[str(ss_list[count][j][1])].append(source_data[s:e])
                        E_dict_target[str(ss_list[count][j][1])].append(target_data[s:e])
                    else:
                        E_dict[str(ss_list[count][j][1])] = [source_data[s:e]]
                        E_dict_target[str(ss_list[count][j][1])] = [target_data[s:e]]
                if ss_list[count][j][0] == 'H':
                    if str(ss_list[count][j][1]) in H_dict.keys():
                        H_dict[str(ss_list[count][j][1])].append(source_data[s:e])
                        H_dict_target[str(ss_list[count][j][1])].append(target_data[s:e])
                    else:
                        H_dict[str(ss_list[count][j][1])] = [source_data[s:e]]
                        H_dict_target[str(ss_list[count][j][1])] = [target_data[s:e]]

                c += ss_list[count][j][1]



        # if count==1:
        #     print(len(C_dict))
        #     print(len(E_dict))
        #     print(len(H_dict))
        #     print(len(C_dict_target))
        #     print(len(E_dict_target))
        #     print(len(H_dict_target))
        #     print(len(C_dict['14']))
        #     print(len(E_dict['10']))
        #     print(len(H_dict['9']))
        #     print(len(C_dict['7']))
        #
        #
        #     print(C_dict['14'][168])
        #     print(E_dict['10'][168])
        #     print(H_dict['9'][168])
        #     print(C_dict['7'][336])
        #     print(C_dict['7'][337])
        #     print('\n')
        #     print(C_dict_target['14'][168])
        #     print(C_dict_target['14'][6])
        #     print(E_dict_target['10'][168])
        #     print(E_dict_target['10'][0])
        #     print(H_dict_target['9'][168])
        #     print(H_dict_target['9'][68])
        #     print(C_dict_target['7'][334])
        #     print(C_dict_target['7'][335])
        #     print(C_dict_target['7'][336])
        #     print(C_dict_target['7'][337])

    f.close()

    json_str = json.dumps(C_dict)
    output_file_C = '../data/partition_data/psi_train/C_source/%s_C_source.json' % target[7:12]
    with open(output_file_C, "w") as json_output:
        json_output.write(json_str)
    json_output.close()

    json_str = json.dumps(E_dict)
    output_file_E = '../data/partition_data/psi_train/E_source/%s_E_source.json' % target[7:12]
    with open(output_file_E, "w") as json_output:
        json_output.write(json_str)
    json_output.close()

    json_str = json.dumps(H_dict)
    output_file_H = '../data/partition_data/psi_train/H_source/%s_H_source.json' % target[7:12]
    with open(output_file_H, "w") as json_output:
        json_output.write(json_str)
    json_output.close()

    json_str = json.dumps(C_dict_target)
    output_file_C = '../data/partition_data/psi_train/C_target/%s_C_target.json' % target[7:12]
    with open(output_file_C, "w") as json_output:
        json_output.write(json_str)
    json_output.close()

    json_str = json.dumps(E_dict_target)
    output_file_E = '../data/partition_data/psi_train/E_target/%s_E_target.json' % target[7:12]
    with open(output_file_E, "w") as json_output:
        json_output.write(json_str)
    json_output.close()

    json_str = json.dumps(H_dict_target)
    output_file_H = '../data/partition_data/psi_train/H_target/%s_H_target.json' % target[7:12]
    with open(output_file_H, "w") as json_output:
        json_output.write(json_str)
    json_output.close()


    count += 1
