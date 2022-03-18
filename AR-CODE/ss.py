from collections import Counter
import json
import os


def generate_ss():
    l = []
    target_path = '../data/CASP12/'
    target_name = os.listdir(target_path)
    for target in target_name:
        with open(target_path + target, 'r') as f:
            dict_data = json.load(f)
            i = 0
            print(len(dict_data))
            for i in range(0, len(dict_data)):
                title = list(dict_data.keys())[i]
                data = dict_data[title]['ss']

                for j in range(0, len(data)):
                    l.append(data[j])
            print(len(data))
        f.close()
    return l

ss=generate_ss()
print(len(ss))
c = Counter(ss)
print(dict(c))
print(len(c))


