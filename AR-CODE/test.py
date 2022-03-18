import os
# def max(a,b,c):
#     if a>=b:
#         max=a
#     else:
#         max=b
#     if max<c:
#         max=c
#     return max


def generate_slices(s, threshold):
    slices=[]
    left = 0
    right = 0
    char_win = list()
    count = {'C': 0, 'E': 0, 'H': 0}
    purity = {'C': 0, 'E': 0, 'H': 0}
    main_elem = s[0]
    while right < len(s):
        char_win.append(s[right])

        if s[right] == "C":
            count['C'] += 1
            purity['C'] = count['C'] / len(char_win)
            purity['E'] = count['E'] / len(char_win)
            purity['H'] = count['H'] / len(char_win)
        if s[right] == "E":
            count['E'] += 1
            purity['C'] = count['C'] / len(char_win)
            purity['E'] = count['E'] / len(char_win)
            purity['H'] = count['H'] / len(char_win)
        if s[right] == "H":
            count['H'] += 1
            purity['C'] = count['C'] / len(char_win)
            purity['E'] = count['E'] / len(char_win)
            purity['H'] = count['H'] / len(char_win)

        # print(purity.values())
        main_elem_purity = max(purity.values())


        if main_elem_purity < threshold:
            while s[right-1] == s[right]:
                right -= 1
            right -= 1
            # print(s[left:right+1])
            slices.append(s[left:right+1])
            count = {'C': 0, 'E': 0, 'H': 0}
            purity = {'C': 0, 'E': 0, 'H': 0}
            char_win.clear()
            right += 1
            left = right
        else:
            right += 1
    slices.append(s[left:right + 1])
    return slices


def filter(slices,minlen):

    i=0
    new_left_purity , new_right_purity=0,0
    while i<len(slices):
        if len(slices)>1:
            if len(slices[i]) < minlen :
                if i == 0:
                    j = len(slices[i])-1
                    while j >= 0:
                        slices[i + 1].insert(0, slices[i][j])
                        j -= 1
                    slices[i].clear()
                    for j in range(i, len(slices) - 1):
                        slices[j] = slices[j + 1]
                    slices.pop()

                elif i == len(slices)-1:
                    for j in range(0, len(slices[i])):
                         slices[i-1].append(slices[i][j])
                    slices.pop()

                else:
                    if len(slices[i-1]) < minlen :
                        if len(slices[i-1]) >= len(slices[i]):
                            left_main_elem_count = len(slices[i-1])
                            left_main_elem = slices[i-1][0]
                        else:
                            left_main_elem_count = len(slices[i])
                            left_main_elem = slices[i][0]
                        new_left_purity = left_main_elem_count / (len(slices[i - 1]) + len(slices[i]))
                    else:
                        left_C_count,left_E_count,left_H_count = 0,0,0
                        for j in range(0,len(slices[i-1])):
                            if slices[i-1][j]== "C":
                                left_C_count += 1
                            if slices[i-1][j]== "E":
                                left_E_count += 1
                            if slices[i-1][j]== "H":
                                left_H_count += 1
                        if left_C_count>=left_E_count:
                            left_main_elem_count = left_C_count
                            left_main_elem = "C"
                        else:
                            left_main_elem_count = left_E_count
                            left_main_elem = "E"
                        if left_H_count>=left_main_elem_count:
                            left_main_elem_count = left_H_count
                            left_main_elem = "H"

                        if slices[i][0] == left_main_elem:
                            new_left_purity = (left_main_elem_count + len(slices[i])) / (len(slices[i-1]) + len(slices[i]))
                        else:
                            new_left_purity = left_main_elem_count  / (len(slices[i - 1]) + len(slices[i]))

                    if len(slices[i + 1]) < minlen:
                        if len(slices[i + 1]) >= len(slices[i]):
                            right_main_elem_count = len(slices[i + 1])
                            right_main_elem = slices[i + 1][0]
                        else:
                            right_main_elem_count = len(slices[i])
                            right_main_elem = slices[i][0]
                        new_right_purity = right_main_elem_count / (len(slices[i + 1]) + len(slices[i]))
                    else:
                        right_C_count, right_E_count, right_H_count = 0, 0, 0
                        for j in range(0, len(slices[i + 1])):
                            if slices[i + 1][j] == "C":
                                right_C_count += 1
                            if slices[i + 1][j] == "E":
                                right_E_count += 1
                            if slices[i + 1][j] == "H":
                                right_H_count += 1
                        if right_C_count >= right_E_count:
                            right_main_elem_count = right_C_count
                            right_main_elem = "C"
                        else:
                            right_main_elem_count = right_E_count
                            right_main_elem = "E"
                        if right_H_count >= right_main_elem_count:
                            right_main_elem_count = right_H_count
                            right_main_elem = "H"
                        if slices[i][0] == right_main_elem:
                            new_right_purity = (right_main_elem_count + len(slices[i])) / (len(slices[i + 1]) + len(slices[i]))
                        else:
                            new_right_purity = right_main_elem_count / (len(slices[i + 1]) + len(slices[i]))

                    if new_left_purity >= new_right_purity:
                        for j in range(0,len(slices[i])):
                            slices[i-1].append(slices[i][j])
                    else:
                        j = len(slices[i])-1
                        while j >= 0:
                            slices[i + 1].insert(0, slices[i][j])
                            j -= 1
                    slices[i].clear()
                    for j in range(i,len(slices)-1):
                        slices[j]=slices[j+1]
                    slices.pop()

            else:
                i += 1
        else:
            break
    return slices


#主元素
def MainElem(s):
    count_C,count_E,count_H=0,0,0
    for i in range(0,len(s)):
        if s[i] == 'C':
            count_C += 1
        elif s[i] == 'E':
            count_E += 1
        else:
            count_H += 1

    if(count_C >= count_E):
        main_elem_count = count_C
        main_elem = 'C'
    else:
        main_elem_count = count_E
        main_elem = 'E'
    if (count_H > main_elem_count):
        main_elem_count = count_H
        main_elem = 'H'
    return main_elem


# 合并相同元素组
def combine_same(slices):
    i = 0
    while i < len(slices)-1:
        if MainElem(slices[i]) == MainElem(slices[i + 1]):
            for j in range(0, len(slices[i+1])):
                slices[i].append(slices[i+1][j])
            slices[i+1].clear()
            for j in range(i+1, len(slices) - 1):
                slices[j] = slices[j + 1]
            slices.pop()
        else:
            i += 1
    return slices


# 返回最终划分结果（对应划分出来的组长及对应主元素，用list保存：[[C,32],[E,25],[H,21],[C,16]]
def results(slices):
    dividlist=[[] for x in range(0,len(slices))]
    for i in range(0,len(slices)):
        dividlist[i].append(MainElem(slices[i]))
        dividlist[i].append(len(slices[i]))
    return dividlist


if __name__ == "__main__":
    # s=["C", "C", "C", "C", "C", "E", "E", "E", "E", "E", "E", "C", "C", "E", "E", "E", "E", "E", "C", "C", "C", "C", "C", "C", "C", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "C", "C", "C", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "C", "C", "C", "C", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "C", "C", "C", "H", "H", "H", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"]
    # # s=["C", "C", "E", "E" , "E", "E", "E", "C", "C", "C", "C", "H", "H", "H", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "E", "E", "E", "C", "C", "C", "C", "E", "E", "E", "C", "C", "C", "C", "H", "H", "H", "C", "C", "C", "C", "C", "C", "C", "C", "C", "E", "E", "E", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "E", "E", "E", "E", "C", "C", "C", "C", "C", "C", "C", "C", "E", "E", "E", "C", "C", "C", "C", "C", "C", "E", "E", "C", "C", "C", "C", "C", "E", "E", "E", "C", "C", "C", "E", "E", "C", "C"]
    # # s=["C", "E", "E", "E", "C", "C", "E", "E", "E", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "E", "E", "C", "C", "E", "E", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "E", "C", "C", "E", "C", "C"]
    # print(s,'\n')
    # slices = generate_slices(s,0.7)
    # print('original partition:',slices,'\n')
    # # slices=[['C', 'H'], ['E', 'E'],['H','C']]
    # slices = filter(slices, 5)
    # print('filter small slices:',slices,'\n')
    # slices=combine_same(slices)
    # print('combine same(finish partition):',slices,'\n')
    # result=results(slices)
    # print('final divide results:', result, '\n')


    ss_list=[]
    target_path = '../data/ss/'
    target_name = os.listdir(target_path)
    # i = 0
    for target in target_name:
        with open(target_path + target, 'r') as f:
            ss_data = f.read()
            print(target)
            print(len(ss_data))
            print(ss_data,'\n')
            ss_data=list(ss_data)
            print('original list:', ss_data, '\n')
            slices = generate_slices(ss_data,0.7)
            print('original partition:',slices,'\n')
            # slices=[['C', 'H'], ['E', 'E'],['H','C']]
            slices = filter(slices, 5)
            print('filter small slices:',slices,'\n')
            slices=combine_same(slices)
            print('combine same(finish partition):',slices,'\n')
            result=results(slices)
            print('final divide results:', result, '\n')
            print('\n')

            ss_list.append(result)
        f.close()
        # i +=1
    print('40 files results:', ss_list,'\n')
    print(len(ss_list))



