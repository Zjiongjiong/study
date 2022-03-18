import sys
import os
from Bio.PDB import *
from Bio.PDB import PDBParser
from Bio.PDB import Superimposer
from Bio.PDB.Atom import *
from Bio.PDB.Residue import *
from Bio.PDB.Chain import *
from Bio.PDB.Model import *
from Bio.PDB.Structure import *
# from Bio.PDB.Vector import *
from Bio.PDB.Entity import*
import math
from PeptideBuilder import Geometry
import PeptideBuilder
import numpy
from os import path
import json

resdict = { 'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', \
	    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', \
	    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', \
	    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y' }

def build_PDB_model(angle_input, PDB_out):
    model_structure_geo=[]
    fh = open(angle_input, "r")
    for line in fh:
        if line[0:1] == "#":
            continue
        tem = line.strip().split()
        if len(tem)!=12:
            print("Warning, the input angles format is not correct, there should be 12 values, please check "+line)
            continue
        try:
            geo = Geometry.geometry(tem[0])
            geo.CA_C_N_angle = float(tem[1])
            geo.C_N_CA_angle = float(tem[2])
            geo.CA_N_length = float(tem[3])
            geo.CA_C_length = float(tem[4])
            geo.peptide_bond = float(tem[5])
            geo.psi_im1 = float(tem[6])
            geo.omega = float(tem[7])
            geo.phi = float(tem[8])
            geo.CA_N_length = float(tem[9])
            geo.CA_C_length = float(tem[10])
            geo.N_CA_C_angle = float(tem[11])
            model_structure_geo.append(geo)
        except:
            print("error format, check line:" + line)
    fh.close()

    outfile = PDBIO()
    outfile.set_structure(PeptideBuilder.make_structure_from_geos(model_structure_geo))
    outfile.save(PDB_out)


if __name__ == "__main__":

    # extract all angles and bond lengths and build PDB
    angles_input_path = '../result/format_pred/'
    angles_input_name = os.listdir(angles_input_path)
    count = 0
    for angles_input in angles_input_name:

        PDB_out = '../result/PDB_out/server_%03d.pdb' % count
        angles_input = angles_input_path + angles_input
        print(count,angles_input,PDB_out)
        build_PDB_model(angles_input, PDB_out)
        count +=1

