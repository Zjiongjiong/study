import json
import numpy as np
import os



def generate_databound(angle_name):
    l = []
    target_path = '../data/CASP12/'
    target_name = os.listdir(target_path)
    for target in target_name:
        with open(target_path + target, 'r') as f:
            dict_data = json.load(f)
            i = 0
            for i in range(0, len(dict_data)):
                title = list(dict_data.keys())[i]
                data = dict_data[title]['Angles'][angle_name]
                for j in range(0, len(data)):
                    l.append(data[j])
        f.close()
    a = np.asarray(l).astype(float)

    return max(a), min(a)


# residue
def load_residue_sequence():
    rseq = [x for x in range(40)]
    count = 0
    target_path = '../data/CASP12/'
    target_name = os.listdir(target_path)
    for target in target_name:
        with open(target_path + target, 'r') as f:
            dict_data = json.load(f)
            title = list(dict_data.keys())[0]
            rseq[count] = dict_data[title]['aa']
            count += 1
        f.close()
    return rseq


def load_length_sequence():
    CA_N_length_seq = [x for x in range(400)]
    CA_C_length_seq = [x for x in range(400)]
    peptide_bond_seq = [x for x in range(400)]
    count = 0
    target_path = '../data/CASP12/'
    target_name = os.listdir(target_path)
    for target in target_name:
        with open(target_path + target, 'r') as f:
            dict_data = json.load(f)
            for i in range(0,10):
                title = list(dict_data.keys())[i]
                CA_N_length_seq[count] = dict_data[title]['Angles']['CA_N_length']
                CA_C_length_seq[count] = dict_data[title]['Angles']['CA_C_length']
                peptide_bond_seq[count] = dict_data[title]['Angles']['peptide_bond']
                count += 1
        f.close()
    return CA_N_length_seq,CA_C_length_seq,peptide_bond_seq
# CA_N_length_seq,CA_C_length_seq,peptide_bond_seq=load_length_sequence()
# print(CA_N_length_seq)


def load_psi_pred():
    psi_pred = [x for x in range(400)]
    count = 0
    target_path = '../result/psi_pred/'
    target_name = os.listdir(target_path)
    for target in target_name:
        with open(target_path + target, 'r') as f:
            dict_data = json.load(f)
            psi_pred[count] = dict_data
            count += 1
        f.close()
    return psi_pred


def load_phi_pred():
    phi_pred = [x for x in range(400)]
    count = 0
    target_path = '../result/phi_pred/'
    target_name = os.listdir(target_path)
    for target in target_name:
        with open(target_path + target, 'r') as f:
            dict_data = json.load(f)
            phi_pred[count] = dict_data
            count += 1
        f.close()
    return phi_pred

def load_omega_pred():
    omega_pred = [x for x in range(400)]
    count = 0
    target_path = '../result/omega_pred/'
    target_name = os.listdir(target_path)
    for target in target_name:
        with open(target_path + target, 'r') as f:
            dict_data = json.load(f)
            omega_pred[count] = dict_data
            count += 1
        f.close()
    return omega_pred


def load_CA_C_N_angle_pred():
    CA_C_N_angle_pred = [x for x in range(400)]
    count = 0
    target_path = '../result/CA_C_N_angle_pred/'
    target_name = os.listdir(target_path)
    for target in target_name:
        with open(target_path + target, 'r') as f:
            dict_data = json.load(f)
            CA_C_N_angle_pred[count] = dict_data
            count += 1
        f.close()
    return CA_C_N_angle_pred


def load_C_N_CA_angle_pred():
    C_N_CA_angle_pred = [x for x in range(400)]
    count = 0
    target_path = '../result/C_N_CA_angle_pred/'
    target_name = os.listdir(target_path)
    for target in target_name:
        with open(target_path + target, 'r') as f:
            dict_data = json.load(f)
            C_N_CA_angle_pred[count] = dict_data
            count += 1
        f.close()
    return C_N_CA_angle_pred


def load_N_CA_C_angle_pred():
    N_CA_C_angle_pred = [x for x in range(400)]
    count = 0
    target_path = '../result/N_CA_C_angle_pred/'
    target_name = os.listdir(target_path)
    for target in target_name:
        with open(target_path + target, 'r') as f:
            dict_data = json.load(f)
            N_CA_C_angle_pred[count] = dict_data
            count += 1
        f.close()
    return N_CA_C_angle_pred


def data_recovery(data,max,min):
    # CA_C_N_angle_pred = load_CA_C_N_angle_pred()
    b = [[] for x in range(400)]
    for i in range(0,len(data)):
        for j in range(0, len(data[i])):
        # a.append(CA_C_N_angle_pred[i])
        # CA_C_N_angle_pred[i] = (CA_C_N_angle_pred[i] / 1000) * (max - min) + min
            data[i][j] = (data[i][j] / 1000) * (max - min) + min
            b[i].append(data[i][j])
    return b


max1,min1=generate_databound('psi_im1')
max2,min2=generate_databound('omega')
max3,min3=generate_databound('phi')
max4,min4=generate_databound('CA_C_N_angle')
max5,min5=generate_databound('C_N_CA_angle')
max6,min6=generate_databound('N_CA_C_angle')
print(max1,min1)
print(max2,min2)
print(max3,min3)
print(max4,min4)
print(max5,min5)
print(max5,min6)

'''
# residue CA_C_N_angle C_N_CA_angle CA_N_length CA_C_length peptide_bond psi_im1 omega phi CA_N_length CA_C_length N_CA_C_angle
for i in range(0,400):
    output_file_path = "../result/format_pred/"
    output_file_name = 'pred_%03d' % i + '.json'
    with open(output_file_path + output_file_name, "w") as json_output:
        residue_sequence = load_residue_sequence()
        CA_C_N_angle_pred=load_CA_C_N_angle_pred()
        CA_C_N_angle_pred =data_recovery(CA_C_N_angle_pred,max4,min4)
        C_N_CA_angle_pred = load_C_N_CA_angle_pred()
        C_N_CA_angle_pred = data_recovery(C_N_CA_angle_pred, max5, min5)
        CA_N_length_seq,CA_C_length_seq,peptide_bond_seq=load_length_sequence()
        psi_pred = load_psi_pred()
        psi_pred = data_recovery(psi_pred, max1, min1)
        omega_pred = load_omega_pred()
        omega_pred = data_recovery(omega_pred, max2, min2)
        phi_pred = load_phi_pred()
        phi_pred = data_recovery(phi_pred, max3, min3)
        N_CA_C_angle_pred = load_N_CA_C_angle_pred()
        N_CA_C_angle_pred = data_recovery(N_CA_C_angle_pred, max6, min6)
        if(i==0):
            print(residue_sequence[0])
            print(residue_sequence[39])
            print(len(residue_sequence))
            print(CA_C_length_seq[0])
            print(CA_C_length_seq[399])
            print(len(CA_C_length_seq))
            print(CA_N_length_seq[0])
            print(CA_N_length_seq[399])
            print(len(CA_N_length_seq))
            print(peptide_bond_seq[0])
            print(peptide_bond_seq[399])
            print(len(peptide_bond_seq))
            print(CA_C_N_angle_pred[0])
            print(CA_C_N_angle_pred[399])
            print(len(CA_C_N_angle_pred))
            print(C_N_CA_angle_pred[0])
            print(C_N_CA_angle_pred[399])
            print(len(C_N_CA_angle_pred))
            print(N_CA_C_angle_pred[0])
            print(N_CA_C_angle_pred[399])
            print(len(N_CA_C_angle_pred))
            print(psi_pred[0])
            print(psi_pred[399])
            print(len(psi_pred))
            print(omega_pred[0])
            print(omega_pred[399])
            print(len(omega_pred))
            print(phi_pred[0])
            print(phi_pred[399])
            print(len(phi_pred))
        print(len(psi_pred[i]))
        json_output.write('# residue CA_C_N_angle C_N_CA_angle CA_N_length CA_C_length peptide_bond psi_im1 omega phi CA_N_length CA_C_length N_CA_C_angle')
        json_output.write('\n')
        for j in range(0,len(psi_pred[i])):

            print(i,j)
            json_output.write(str(residue_sequence[int(i/10)][j]))
            json_output.write(' ')
            json_output.write(str(CA_C_N_angle_pred[i][j]))
            json_output.write(' ')
            json_output.write(str(C_N_CA_angle_pred[i][j]))
            json_output.write(' ')
            json_output.write(str(CA_N_length_seq[i][j]))
            json_output.write(' ')
            json_output.write(str(CA_C_length_seq[i][j]))
            json_output.write(' ')
            json_output.write(str(peptide_bond_seq[i][j]))
            json_output.write(' ')
            json_output.write(str(psi_pred[i][j]))
            json_output.write(' ')
            json_output.write(str(omega_pred[i][j]))
            json_output.write(' ')
            json_output.write(str(phi_pred[i][j]))
            json_output.write(' ')
            json_output.write(str(CA_N_length_seq[i][j]))
            json_output.write(' ')
            json_output.write(str(CA_C_length_seq[i][j]))
            json_output.write(' ')
            json_output.write(str(N_CA_C_angle_pred[i][j]))
            json_output.write(' ')
            json_output.write('\n')

    json_output.close()
'''
# residue CA_C_N_angle C_N_CA_angle CA_N_length CA_C_length peptide_bond psi_im1 omega phi CA_N_length CA_C_length N_CA_C_angle
residue_sequence = load_residue_sequence()
CA_C_N_angle_pred=load_CA_C_N_angle_pred()
CA_C_N_angle_pred =data_recovery(CA_C_N_angle_pred,max4,min4)
C_N_CA_angle_pred = load_C_N_CA_angle_pred()
C_N_CA_angle_pred = data_recovery(C_N_CA_angle_pred, max5, min5)
CA_N_length_seq,CA_C_length_seq,peptide_bond_seq=load_length_sequence()
psi_pred = load_psi_pred()
psi_pred = data_recovery(psi_pred, max1, min1)
omega_pred = load_omega_pred()
omega_pred = data_recovery(omega_pred, max2, min2)
phi_pred = load_phi_pred()
phi_pred = data_recovery(phi_pred, max3, min3)
N_CA_C_angle_pred = load_N_CA_C_angle_pred()
N_CA_C_angle_pred = data_recovery(N_CA_C_angle_pred, max6, min6)

print(residue_sequence[0])
print(residue_sequence[39])
print(len(residue_sequence))
print(CA_C_length_seq[0])
print(CA_C_length_seq[399])
print(len(CA_C_length_seq))
print(CA_N_length_seq[0])
print(CA_N_length_seq[399])
print(len(CA_N_length_seq))
print(peptide_bond_seq[0])
print(peptide_bond_seq[399])
print(len(peptide_bond_seq))
print(CA_C_N_angle_pred[0])
print(CA_C_N_angle_pred[399])
print(len(CA_C_N_angle_pred))
print(C_N_CA_angle_pred[0])
print(C_N_CA_angle_pred[399])
print(len(C_N_CA_angle_pred))
print(N_CA_C_angle_pred[0])
print(N_CA_C_angle_pred[399])
print(len(N_CA_C_angle_pred))
print(psi_pred[0])
print(psi_pred[399])
print(len(psi_pred))
print(omega_pred[0])
print(omega_pred[399])
print(len(omega_pred))
print(phi_pred[0])
print(phi_pred[399])
print(len(phi_pred))

for i in range(0,400):
    list = []
    output_file_path = "../result/format_pred/"
    output_file_name = 'pred_%03d' % i + '.json'
    with open(output_file_path + output_file_name, "w") as json_output:
        print(len(psi_pred[i]))
        json_output.write('# residue CA_C_N_angle C_N_CA_angle CA_N_length CA_C_length peptide_bond psi_im1 omega phi CA_N_length CA_C_length N_CA_C_angle')
        json_output.write('\n')

        for j in range(0,len(psi_pred[i])):

            print(i,j)
            '''
            json_output.write(str(residue_sequence[int(i/10)][j]))
            json_output.write(' ')
            json_output.write(str(CA_C_N_angle_pred[i][j]))
            json_output.write(' ')
            json_output.write(str(C_N_CA_angle_pred[i][j]))
            json_output.write(' ')
            json_output.write(str(CA_N_length_seq[i][j]))
            json_output.write(' ')
            json_output.write(str(CA_C_length_seq[i][j]))
            json_output.write(' ')
            json_output.write(str(peptide_bond_seq[i][j]))
            json_output.write(' ')
            json_output.write(str(psi_pred[i][j]))
            json_output.write(' ')
            json_output.write(str(omega_pred[i][j]))
            json_output.write(' ')
            json_output.write(str(phi_pred[i][j]))
            json_output.write(' ')
            json_output.write(str(CA_N_length_seq[i][j]))
            json_output.write(' ')
            json_output.write(str(CA_C_length_seq[i][j]))
            json_output.write(' ')
            json_output.write(str(N_CA_C_angle_pred[i][j]))
            json_output.write(' ')
            json_output.write('\n')
            '''
            list.append(str(residue_sequence[int(i / 10)][j]) + ' ')
            list.append(str(CA_C_N_angle_pred[i][j]) + ' ')
            list.append(str(C_N_CA_angle_pred[i][j]) + ' ')
            list.append(str(CA_N_length_seq[i][j]) + ' ')
            list.append(str(CA_C_length_seq[i][j]) + ' ')
            list.append(str(peptide_bond_seq[i][j]) + ' ')
            list.append(str(psi_pred[i][j]) + ' ')
            list.append(str(omega_pred[i][j]) + ' ')
            list.append(str(phi_pred[i][j]) + ' ')
            list.append(str(CA_N_length_seq[i][j]) + ' ')
            list.append(str(CA_C_length_seq[i][j]) + ' ')
            list.append(str(N_CA_C_angle_pred[i][j]) + ' ')
            list.append('\n')
        json_output.writelines(list)
    json_output.close()