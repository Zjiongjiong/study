import os
import linecache
import json

# def run_GDT(model,native):
#     cmd = ' TMscore ' + model + native
#     os.system(cmd)


def run_GDT(model,native):
    cmd = ' TMscore ' + model + ' ' + native + '>' + GDT_out
    os.system(cmd)


def load_original_GDT():
    GDT_list = []
    total_GDT = 0
    target_path = '../data/CASP12/'
    target_name = os.listdir(target_path)
    for target in target_name:
        with open(target_path + target, 'r') as f:
            dict_data = json.load(f)
            for i in range(0, 10):
                title = list(dict_data.keys())[i]
                GDT_list.append(float(dict_data[title]['GDT']))
                total_GDT +=float(dict_data[title]['GDT'])
        f.close()
    return GDT_list,total_GDT/len(GDT_list)

if __name__ == "__main__":
    '''
    model = ' ../data/model/server01.pdb '
    native = ' ../data/native/T0859.pdb '
    GDT_out = ' ../data/GDT_out/pred.out '
    run_GDT(model,native)   
    '''

    score_list = []
    model_path = '../data/model/'
    model_name = os.listdir(model_path)
    count = 0
    total_score=0
    for model in model_name:
        with open(model_path + model, 'r') as f:
            native_path = '../data/native/'
            native_name = os.listdir(native_path)
            native = native_name[int(count / 10)]
            native = native_path+native

            model = model_path + model

            GDT_out = '../data/GDT_out/pred_%03d.out'%count
            print('Native:',native,'    Model:',model,'     GDT_out:',GDT_out)
            run_GDT(model, native)
            GDT_score = linecache.getline(GDT_out, 19)[14:20]
            print(GDT_score+'\n')
            if(len(GDT_score)==0):
                GDT_score=0
            score_list.append(float(GDT_score))
            total_score +=float(GDT_score)
        f.close()
        count += 1
    print(count)
    print(score_list)

    print('Average GDT SCORE:\n')
    GDT_list, avg_GDT = load_original_GDT()
    print('Before:  %f' % avg_GDT)
    print(score_list)
    print('After:   %f' % (total_score/len(score_list)))
    print(GDT_list)





