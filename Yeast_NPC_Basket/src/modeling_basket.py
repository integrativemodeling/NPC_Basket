#!/usr/bin/env python
#####################################################
# Last Update: March 26, 2024 by Neelesh Soni
# Andrej Sali group, University of California San Francisco (UCSF)
#####################################################

# --------------------------------------------------------------
# Import IMP modules
# --------------------------------------------------------------
import IMP
import IMP.atom
import IMP.core
import IMP.bayesianem
import IMP.bayesianem.restraint
import IMP.rmf
import IMP.pmi
import IMP.pmi.topology
import IMP.pmi.dof
import IMP.pmi.macros
import IMP.pmi.io
import IMP.pmi.io.crosslink
import IMP.pmi.restraints
import IMP.pmi.restraints.em
import IMP.pmi.restraints.stereochemistry
import IMP.pmi.restraints.crosslinking
import IMP.pmi.restraints.npc
import ihm.cross_linkers
import IMP.pmi.restraints.basic
from IMP.pmi.io.crosslink import FilterOperator as FO

# --------------------------------------------------------------
# Import other python modules
# --------------------------------------------------------------
import time
import sys
import os
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import operator

from collections import defaultdict
from argparse import ArgumentParser, RawDescriptionHelpFormatter


# --------------------------------------------------------------
# Coiled-COil Segment Definitions
# --------------------------------------------------------------

# Extract coiled-coil segments from a COCONUT alignment file. This involves reading the
# file, parsing the defined segments, and returning a list of these segments.

def read_alignment_file(fname):
	"""Summary
	
	Args:
	    fname (TYPE): Coiled-coil segments alignment file
	
	Returns:
	    TYPE: Coiled-coil segments list of list
	"""

	inf = open(fname,'r')
	lines = inf.readlines()
	inf.close()

	cc_segments=[]

	for l in lines:
		
		if ((l[0]=='-') or (l[0]=='#') or len(l)==1):
			continue

		toks = l.strip().split('\t')
		
		#Segments for protein 1
		respair1 = list(map(int,toks[1].split('-')))

		#Segments for protein 2
		respair2 = list(map(int,toks[3].split('-')))

		cc_segments.append([ respair1, respair2 ])

	return cc_segments


# --------------------------------------------------------------
# Command Line arguments Definitions
# --------------------------------------------------------------

parser_obj = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)

parser_obj.add_argument("-o", "--output_dir", default="output", help="Output files directory")

parser_obj.add_argument("-d", "--data_dir", default="data", help="Data files directory")

args = parser_obj.parse_args()
data_path = os.path.abspath(args.data_dir)
out_path = os.path.abspath(args.output_dir)

print("Following Command Line Argumenst were used: \n")
print("data path:",data_path)
print("output path:",out_path)

# --------------------------------------------------------------
# Parameters for the Simulations
# --------------------------------------------------------------

include_crosslink_restraint1=True
include_crosslink_restraint2=True
include_gaussian_em_restraint=True
include_membrane_surface_binding=True
include_zaxial_position_restraint=True
include_membrane_surface_binding=True
include_zaxial_position_restraint=True


#Contains all Bssket xlinks and ONLY Inter Ycom-Bskt xlinks. NO Intra-Ycom xlinks
xlinkfilename_newDSS = os.path.join(data_path,"xlinks/xlinks_InterYcom_Bskt_newDSS.csv")
xlinkfilename_other = os.path.join(data_path,"xlinks/xlinks_InterYcom_Bskt_others.csv")  

#Load GMM definitions
gmmfilename = os.path.join(data_path,'gmms/Yeast_1_BR_v4_center_ng80.txt')

#Load pdb models
mlps_pdb_fn=os.path.join(data_path,'pdbs/MLP_MLP.pdb')
nup1_pdb_fn=os.path.join(data_path,'pdbs/Nup1.pdb')
nup2_pdb_fn=os.path.join(data_path,'pdbs/Nup2.pdb')
nup60_pdb_fn=os.path.join(data_path,'pdbs/Nup60.pdb')


# Load FASTA files
alifname = os.path.join(data_path,'alignments/MLP_MLP_alignment.cali')
mlp1_fastafile = os.path.join(data_path,'fasta/MLP.fasta')
mlp2_fastafile = os.path.join(data_path,'fasta/MLP.fasta')

nup1_fastafile = os.path.join(data_path,'fasta/Nup1.fasta')
nup2_fastafile = os.path.join(data_path,'fasta/Nup2.fasta')
nup60_fastafile = os.path.join(data_path,'fasta/Nup60.fasta')


EV_RESOLUTION=30
INIT_TRANS = 500.0

# rigid body/beads params
beadsize_0 = 1
beadsize_1 = 5 #For high resoluton particles.
beadsize_2 = 30 #For medium resolution particles
beadsize_3 = 30 #For medium resolution particles
beadsize_4 = 50 #For low resolution particles

MC_TRANS = 10.0
MC_ROT   = 0.5 * (22/7)

# XL restraint params
XL_CUT = 30.0
XL_SLOPE = 0.02
XL_WEIGHT = 0.1
XL_WEIGHT1 = 1.0

EM_SLOPE = 0.1
MSLR_WEIGHT = 1.0
EMR_WEIGHT = 1.0

# replica exchange params
MIN_REX_TEMP = 1.0
MAX_REX_TEMP = 4.0

optimize_flex_beads_steps=50

MC_STEPS_PER_FRAME = 20
NUM_FRAMES_1 = 1000
NUM_FRAMES_2 = 100000


#CLONES CHAINS IDs
#NO CLONES AT ALL
chain_list = {'A':'','a':''}

mlp1_nterm = [1,70] #all inclusive
mlp1_cterm = [1458,1875] #all inclusive
mlp1_nbs_domain = [338,616] 
mlp1_nbs_domain_loop = [488,530] #For z-axial restraint for NBS domain

mlp2_nterm = [1,70] #all inclusive
mlp2_cterm = [1458,1875] #all inclusive

mlp1_spk1_spk3_upperdist=50
mlp2_spk1_spk3_upperdist=50

#First HELIX FOR MEMBRANE Restraint only, all others for structural restraint
nup1_helix_res=[[1,32],[85,104],[106,123]]
nup1_FGRepeat = [336,1076] #CTerm
nup1_rigid_bodies = [[1,32],[85,104],[106,123]]

nup2_FGRepeat = [137,582] #CTerm
nup2_ExtraRemoved1 = [1,50] #Disordered and no interaction
nup2_ExtraRemoved2 = [583,601] #Disordered and no interaction
nup2_rigid_bodies = [[83,136], [602,720]]

#https://www.pnas.org/doi/full/10.1073/pnas.0403643101#sec-1
#Rg=R0*N^v  where R0=1.330 and v=0.598 at 95% confidence interval
#Using above formula for upperbound distance btw disordered regions. d = 2*Rg
#Since, FG Repeats (that remains near the central channel) is between the 
#two domains (that remains near structural regions or NPC periphery),
# and NPC 8 fold symmetry, theta=45, the two nup2 domains has to be restrained at d=2Rgsin(theta/2) distance
#nup2_structural_domain_scaling = 20.0
nup2_disordered_len = (601 - 137 + 1)
nup2_upper_dist = (1.33*np.power(nup2_disordered_len,0.598)) * 2 * np.sin(2*np.pi/16) #This comes aorund 40A


nup60_helix_res = [[27,47],[91,104],[106,119],[121,140],[142,162]]
# Not considering nup60 loop region (463-476) that Xls with nup2.
nup60_FGRepeat = [[399,504]]
nup60_disordered_len = 504 - 399 + 1
nup60_upper_dist = (1.33*np.power(nup60_disordered_len,0.598)) * 2  
nup60_rigid_bodies = [[27,47],[91,104],[106,119],[121,140],[142,162]]

#Interactiing domain for residue proximity restraint
nup60_n2bm_domain = [505,539]
nup2_n60bm_domain =[83,136]
n60bm_n2bm_upperdist = 10.0

bskt_mols =[]
bskt_nup_mols = []
ydimer_mols=[]


# --------------------------------------------------------------
# Read Fasta Files
# --------------------------------------------------------------

mlp1_seqs = IMP.pmi.topology.Sequences(mlp1_fastafile)
mlp2_seqs = IMP.pmi.topology.Sequences(mlp2_fastafile)
nup1_seqs = IMP.pmi.topology.Sequences(nup1_fastafile)
nup2_seqs = IMP.pmi.topology.Sequences(nup2_fastafile)
nup60_seqs = IMP.pmi.topology.Sequences(nup60_fastafile)

#Color map iterator in the Chimera format
Rcolor1 = cm.get_cmap('Set3', 12) #No of colors in Set3 = 12
Rcolor2 = cm.get_cmap('Pastel1', 8) #No of colors in Set2 = 8
Rcolors = iter(np.vstack((Rcolor1(np.linspace(0,1,12))[:,0:3],Rcolor2(np.linspace(0,1,8))[:,0:3])))


# Create System and State
mdl = IMP.Model()
s = IMP.pmi.topology.System(mdl)
st = s.create_state()


ydimer_fastafile = os.path.join(data_path,'fasta/Y-dimer.fasta')
Ydimer_seqs = IMP.pmi.topology.Sequences(ydimer_fastafile)
ydimer = os.path.join(data_path,'pdbs/Y-dimer-modified-aligned.pdb')

YDIMER={'Nup120':['A','L'],'Nup85':['B','M','H'],'Nup145c':['C','N'],
		'Sec13':['D','O'],'Seh1':['E','P'],'Nup84':['F','Q'],
		'Nup133':['G','R']}


# --------------------------------------------------------------
# MLP molecule
# --------------------------------------------------------------

#Get cc segments from the alignment file
cc_segments = read_alignment_file(alifname)

rcolor = next(Rcolors);bcolor = rcolor
mlp1_mol1 = st.create_molecule('mlp', sequence=mlp1_seqs['MLP'+':A'], chain_id='A')
mlp1_atomic1 = mlp1_mol1.add_structure(mlps_pdb_fn, chain_id='A',offset=0, ca_only = True)

for i, ccseg in enumerate(list(cc_segments)):
	mlp1_mol1.add_representation(mlp1_mol1[ccseg[0][0]-1:ccseg[0][1]],
		resolutions=[beadsize_0,beadsize_3], color=rcolor,
		density_residues_per_component=20,
		density_prefix=out_path+'/mlp1_gmm'+str(i),
		density_force_compute=False,
		density_voxel_size=10.0)

mlp1_mol1_str = mlp1_mol1.get_atomic_residues()
mlp1_mol1_loop = mlp1_mol1.get_non_atomic_residues()
mlp1_mol1_Nterm = mlp1_mol1[mlp1_nterm[0]-1:mlp1_nterm[1]]
mlp1_mol1_Cterm = mlp1_mol1[mlp1_cterm[0]-1:mlp1_cterm[1]]
mlp1_mol1_nbs = mlp1_mol1[mlp1_nbs_domain[0]-1:mlp1_nbs_domain[1]]


#Add one resolution for un-Structured regions
mlp1_mol1.add_representation(mlp1_mol1_loop-mlp1_mol1_Nterm-mlp1_mol1_Cterm,
	resolutions=[beadsize_1], color=bcolor,
	setup_particles_as_densities=True)

#Add one resolution for un-Structured regions
mlp1_mol1.add_representation(mlp1_mol1_Nterm,
	resolutions=[beadsize_4], color=bcolor,
	setup_particles_as_densities=True)

#Add one resolution for un-Structured regions
mlp1_mol1.add_representation(mlp1_mol1_Cterm,
	resolutions=[beadsize_4], color=bcolor,
	setup_particles_as_densities=True)

bskt_mols.append(mlp1_mol1)


rcolor = next(Rcolors);bcolor = rcolor
mlp2_mol1 = mlp1_mol1.create_copy( chain_id='a')
mlp2_atomic1 = mlp2_mol1.add_structure(mlps_pdb_fn, chain_id='B',offset=0,ca_only = True)

for i, ccseg in enumerate(list(cc_segments)):

	mlp2_mol1.add_representation(mlp2_mol1[ccseg[1][0]-1:ccseg[1][1]],
		resolutions=[beadsize_0,beadsize_3], color=rcolor,
		density_residues_per_component=20,
		density_prefix=out_path+'/mlp2_gmm'+str(i),
		density_force_compute=False,
		density_voxel_size=10.0)

mlp2_mol1_str = mlp2_mol1.get_atomic_residues()
mlp2_mol1_loop = mlp2_mol1.get_non_atomic_residues()
mlp2_mol1_Nterm = mlp2_mol1[mlp2_nterm[0]-1:mlp2_nterm[1]]
mlp2_mol1_Cterm = mlp2_mol1[mlp2_cterm[0]-1:mlp2_cterm[1]]


#Add one resolution for un-Structured regions 
mlp2_mol1.add_representation(mlp2_mol1_loop-mlp2_mol1_Nterm-mlp2_mol1_Cterm,
	resolutions=[beadsize_1], color=bcolor,
	setup_particles_as_densities=True)

#Add one resolution for un-Structured regions 
mlp2_mol1.add_representation(mlp2_mol1_Nterm,
	resolutions=[beadsize_4], color=bcolor,
	setup_particles_as_densities=True)

#Add one resolution for un-Structured regions 
mlp2_mol1.add_representation(mlp2_mol1_Cterm,
	resolutions=[beadsize_4], color=bcolor,
	setup_particles_as_densities=True)

bskt_mols.append(mlp2_mol1)


# --------------------------------------------------------------
# NUP1 molecule
# --------------------------------------------------------------

rcolor = next(Rcolors);bcolor = rcolor 
Nup1_mol1 = st.create_molecule('nup1', sequence=nup1_seqs['NUP1'+':A'], chain_id='A')
Nup1_atomic1 = Nup1_mol1.add_structure(nup1_pdb_fn, chain_id='A',offset=0,ca_only = True)

for i, rgbhelices in enumerate(list(nup1_rigid_bodies)):

	Nup1_mol1.add_representation(
		Nup1_mol1[rgbhelices[0]-1:rgbhelices[1]],
		resolutions=[beadsize_0,beadsize_3], color=bcolor)

Nup1_mol1_str = Nup1_mol1.get_atomic_residues()

Nup1_mol1_loop = Nup1_mol1[:] - Nup1_mol1_str \
	- Nup1_mol1[nup1_FGRepeat[0]-1:nup1_FGRepeat[1]] #FG repeats

Nup1_mol1.add_representation(
	Nup1_mol1_loop,
	resolutions=[beadsize_2], color=bcolor)

bskt_nup_mols.append(Nup1_mol1)
bskt_mols.append(Nup1_mol1)


# --------------------------------------------------------------
# NUP2 molecule 0
# --------------------------------------------------------------

rcolor = next(Rcolors);bcolor = rcolor
Nup2_mol0 = st.create_molecule('nup2', sequence=nup2_seqs['NUP2'+':A'], chain_id='A')
Nup2_atomic0 = Nup2_mol0.add_structure(nup2_pdb_fn, chain_id='A',offset=0,ca_only = True)

Nup2_mol0_str = Nup2_mol0.get_atomic_residues()

Nup2_mol0_loop = Nup2_mol0[:] \
	- Nup2_mol0_str \
	- Nup2_mol0[nup2_FGRepeat[0]-1:nup2_FGRepeat[1]] \
	- Nup2_mol0[nup2_ExtraRemoved1[0]-1:nup2_ExtraRemoved1[1]] \
	- Nup2_mol0[nup2_ExtraRemoved2[0]-1:nup2_ExtraRemoved2[1]]

Nup2_mol0.add_representation(Nup2_mol0_str,
	resolutions=[beadsize_0,beadsize_3], color=rcolor)


Nup2_mol0.add_representation(
	Nup2_mol0_loop,
	resolutions=[beadsize_2], color=bcolor)


bskt_nup_mols.append(Nup2_mol0)
bskt_mols.append(Nup2_mol0)

# --------------------------------------------------------------
# NUP2 molecule 1 (copy of previous NUP2)
# --------------------------------------------------------------

rcolor = next(Rcolors);bcolor = rcolor 
Nup2_mol1 = Nup2_mol0.create_copy( chain_id='a')
Nup2_atomic1 = Nup2_mol1.add_structure(nup2_pdb_fn, chain_id='A',offset=0,ca_only = True)

Nup2_mol1_str = Nup2_mol1.get_atomic_residues()

Nup2_mol1_loop = Nup2_mol1[:] \
	- Nup2_mol1_str \
	- Nup2_mol1[nup2_FGRepeat[0]-1:nup2_FGRepeat[1]] \
	- Nup2_mol1[nup2_ExtraRemoved1[0]-1:nup2_ExtraRemoved1[1]] \
	- Nup2_mol1[nup2_ExtraRemoved2[0]-1:nup2_ExtraRemoved2[1]]

Nup2_mol1.add_representation(Nup2_mol1_str,
	resolutions=[beadsize_0,beadsize_3], color=rcolor)


Nup2_mol1.add_representation(
	Nup2_mol1_loop,
	resolutions=[beadsize_2], color=bcolor)


bskt_nup_mols.append(Nup2_mol1)
bskt_mols.append(Nup2_mol1)


# --------------------------------------------------------------
# NUP60 molecule 0
# --------------------------------------------------------------

rcolor = next(Rcolors);bcolor = rcolor
Nup60_mol0 = st.create_molecule('nup60', sequence=nup60_seqs['NUP60'+':A'], chain_id='A')
Nup60_atomic0 = Nup60_mol0.add_structure(nup60_pdb_fn, chain_id='A',offset=0,ca_only = True)

for i, rgbhelices in enumerate(list(nup60_rigid_bodies)):

	Nup60_mol0.add_representation(
		Nup60_mol0[rgbhelices[0]-1:rgbhelices[1]],
		resolutions=[beadsize_0,beadsize_3], color=bcolor)

Nup60_mol0_str = Nup60_mol0.get_atomic_residues()

Nup60_mol0_loop = Nup60_mol0[:] \
	- Nup60_mol0_str \
	- Nup60_mol0[nup60_FGRepeat[0][0]-1:nup60_FGRepeat[0][1]]


Nup60_mol0.add_representation(
	Nup60_mol0_loop
	,resolutions=[beadsize_2], color=bcolor)


bskt_nup_mols.append(Nup60_mol0)
bskt_mols.append(Nup60_mol0)


# --------------------------------------------------------------
# NUP60 molecule 1 (copy of previous NUP60)
# --------------------------------------------------------------

rcolor = next(Rcolors);bcolor = rcolor
Nup60_mol1 = Nup60_mol0.create_copy( chain_id='a')
Nup60_atomic1 = Nup60_mol1.add_structure(nup60_pdb_fn, chain_id='A',offset=0,ca_only = True)

for i, rgbhelices in enumerate(list(nup60_rigid_bodies)):

	Nup60_mol1.add_representation(
		Nup60_mol1[rgbhelices[0]-1:rgbhelices[1]],
		resolutions=[beadsize_0,beadsize_3], color=bcolor)

Nup60_mol1_str = Nup60_mol1.get_atomic_residues()

Nup60_mol1_loop = Nup60_mol1[:] \
	- Nup60_mol1_str \
	- Nup60_mol1[nup60_FGRepeat[0][0]-1:nup60_FGRepeat[0][1]]


Nup60_mol1.add_representation(
	Nup60_mol1_loop
	,resolutions=[beadsize_2], color=bcolor)


bskt_nup_mols.append(Nup60_mol1)
bskt_mols.append(Nup60_mol1)

# --------------------------------------------------------------
# Y-Dimer complex as rigid body, no representation for mising structures
# --------------------------------------------------------------

rcolor = next(Rcolors);bcolor = rcolor
Nup120_mol1 = st.create_molecule('nup120', sequence=Ydimer_seqs['Nup120'], chain_id='A')
Nup120_atomic1 = Nup120_mol1.add_structure(ydimer, chain_id=YDIMER['Nup120'][0],offset=0,ca_only = True)
Nup120_mol1.add_representation(Nup120_atomic1,resolutions=[beadsize_0,beadsize_3], color=rcolor)
Nup120_mol1.add_representation(Nup120_mol1[:]-Nup120_atomic1,resolutions=[beadsize_4], color=bcolor)
ydimer_mols.append(Nup120_mol1)


Nup120_mol2 = Nup120_mol1.create_copy( chain_id='a')
Nup120_atomic2 = Nup120_mol2.add_structure(ydimer, chain_id=YDIMER['Nup120'][1],offset=0,ca_only = True)
Nup120_mol2.add_representation(Nup120_atomic2,resolutions=[beadsize_0,beadsize_3], color=rcolor)
Nup120_mol2.add_representation(Nup120_mol2[:]-Nup120_atomic2,resolutions=[beadsize_4], color=bcolor)
ydimer_mols.append(Nup120_mol2)


rcolor = next(Rcolors);bcolor = rcolor 
Nup85_mol1 = st.create_molecule('nup85', sequence=Ydimer_seqs['Nup85'], chain_id='A')
Nup85_atomic1 = Nup85_mol1.add_structure(ydimer, chain_id=YDIMER['Nup85'][0],offset=0,ca_only = True)
Nup85_mol1.add_representation(Nup85_atomic1,resolutions=[beadsize_0,beadsize_3], color=rcolor)
Nup85_mol1.add_representation(Nup85_mol1[:]-Nup85_atomic1,resolutions=[beadsize_4], color=bcolor)
ydimer_mols.append(Nup85_mol1)


Nup85_mol2 = Nup85_mol1.create_copy( chain_id='a')
Nup85_atomic2 = Nup85_mol2.add_structure(ydimer, chain_id=YDIMER['Nup85'][1],offset=0,ca_only = True)
Nup85_mol2.add_representation(Nup85_atomic2,resolutions=[beadsize_0,beadsize_3], color=rcolor)
Nup85_mol2.add_representation(Nup85_mol2[:]-Nup85_atomic2,resolutions=[beadsize_4], color=bcolor)
ydimer_mols.append(Nup85_mol2)


Nup85_mol3 = Nup85_mol1.create_copy( chain_id='b')
Nup85_atomic3 = Nup85_mol3.add_structure(ydimer, chain_id=YDIMER['Nup85'][2],offset=0,ca_only = True)
Nup85_mol3.add_representation(Nup85_atomic3,resolutions=[beadsize_0,beadsize_3], color=rcolor)
Nup85_mol3.add_representation(Nup85_mol3[:]-Nup85_atomic3,resolutions=[beadsize_1], color=bcolor) #This forms cross-link thus high resolution for this domain
ydimer_mols.append(Nup85_mol3)


rcolor = next(Rcolors);bcolor = rcolor 
Nup145c_mol1 = st.create_molecule('nup145c', sequence=Ydimer_seqs['Nup145c'], chain_id='A')
Nup145c_atomic1 = Nup145c_mol1.add_structure(ydimer, chain_id=YDIMER['Nup145c'][0],offset=0,ca_only = True)
Nup145c_mol1.add_representation(Nup145c_atomic1,resolutions=[beadsize_0,beadsize_3], color=rcolor)
Nup145c_mol1.add_representation(Nup145c_mol1[:]-Nup145c_atomic1,resolutions=[beadsize_4], color=bcolor)
ydimer_mols.append(Nup145c_mol1)


Nup145c_mol2 = Nup145c_mol1.create_copy(chain_id='a')
Nup145c_atomic2 = Nup145c_mol2.add_structure(ydimer, chain_id=YDIMER['Nup145c'][1],offset=0,ca_only = True)
Nup145c_mol2.add_representation(Nup145c_atomic2,resolutions=[beadsize_0,beadsize_3], color=rcolor)
Nup145c_mol2.add_representation(Nup145c_mol2[:]-Nup145c_atomic2,resolutions=[beadsize_4], color=bcolor)
ydimer_mols.append(Nup145c_mol2)


rcolor = next(Rcolors);bcolor = rcolor 
Sec13_mol1 = st.create_molecule('sec13', sequence=Ydimer_seqs['Sec13'], chain_id='A')
Sec13_atomic1 = Sec13_mol1.add_structure(ydimer, chain_id=YDIMER['Sec13'][0],offset=0,ca_only = True)
Sec13_mol1.add_representation(Sec13_atomic1,resolutions=[beadsize_0,beadsize_3], color=rcolor)
Sec13_mol1.add_representation(Sec13_mol1[:]-Sec13_atomic1,resolutions=[beadsize_4], color=bcolor)
ydimer_mols.append(Sec13_mol1)


Sec13_mol2 = Sec13_mol1.create_copy(chain_id='a')
Sec13_atomic2 = Sec13_mol2.add_structure(ydimer, chain_id=YDIMER['Sec13'][1],offset=0,ca_only = True)
Sec13_mol2.add_representation(Sec13_atomic2,resolutions=[beadsize_0,beadsize_3], color=rcolor)
Sec13_mol2.add_representation(Sec13_mol2[:]-Sec13_atomic2,resolutions=[beadsize_4], color=bcolor)
ydimer_mols.append(Sec13_mol2)


rcolor = next(Rcolors);bcolor = rcolor 
Seh1_mol1 = st.create_molecule('seh1', sequence=Ydimer_seqs['Seh1'], chain_id='A')
Seh1_atomic1 = Seh1_mol1.add_structure(ydimer, chain_id=YDIMER['Seh1'][0],offset=0,ca_only = True)
Seh1_mol1.add_representation(Seh1_atomic1,resolutions=[beadsize_0,beadsize_3], color=rcolor)
Seh1_mol1.add_representation(Seh1_mol1[:]-Seh1_atomic1,resolutions=[beadsize_4], color=bcolor)
ydimer_mols.append(Seh1_mol1)


Seh1_mol2 = Seh1_mol1.create_copy(chain_id='a')
Seh1_atomic2 = Seh1_mol2.add_structure(ydimer, chain_id=YDIMER['Seh1'][1],offset=0,ca_only = True)
Seh1_mol2.add_representation(Seh1_atomic2,resolutions=[beadsize_0,beadsize_3], color=rcolor)
Seh1_mol2.add_representation(Seh1_mol2[:]-Seh1_atomic2,resolutions=[beadsize_4], color=bcolor)
ydimer_mols.append(Seh1_mol2)


rcolor = next(Rcolors);bcolor = rcolor 
Nup84_mol1 = st.create_molecule('nup84', sequence=Ydimer_seqs['Nup84'], chain_id='A')
Nup84_atomic1 = Nup84_mol1.add_structure(ydimer, chain_id=YDIMER['Nup84'][0][0],offset=0,ca_only = True)
Nup84_mol1.add_representation(Nup84_atomic1,resolutions=[beadsize_0,beadsize_3], color=rcolor)
Nup84_mol1.add_representation(Nup84_mol1[:]-Nup84_atomic1,resolutions=[beadsize_4], color=bcolor)
ydimer_mols.append(Nup84_mol1)


Nup84_mol2 = Nup84_mol1.create_copy( chain_id='a')
Nup84_atomic2 = Nup84_mol2.add_structure(ydimer, chain_id=YDIMER['Nup84'][1][0],offset=0,ca_only = True)
Nup84_mol2.add_representation(Nup84_atomic2,resolutions=[beadsize_0,beadsize_3], color=rcolor)
Nup84_mol2.add_representation(Nup84_mol2[:]-Nup84_atomic2,resolutions=[beadsize_4], color=bcolor)
ydimer_mols.append(Nup84_mol2)


rcolor = next(Rcolors);bcolor = rcolor 
Nup133_mol1 = st.create_molecule('nup133', sequence=Ydimer_seqs['Nup133'], chain_id='A')
Nup133_atomic1 = Nup133_mol1.add_structure(ydimer, chain_id=YDIMER['Nup133'][0][0],offset=0,ca_only = True)
Nup133_mol1.add_representation(Nup133_atomic1,resolutions=[beadsize_0,beadsize_3], color=rcolor)
Nup133_mol1.add_representation(Nup133_mol1[:]-Nup133_atomic1,resolutions=[beadsize_4], color=bcolor)
ydimer_mols.append(Nup133_mol1)


Nup133_mol2 = Nup133_mol1.create_copy(chain_id='a')
Nup133_atomic2 = Nup133_mol2.add_structure(ydimer, chain_id=YDIMER['Nup133'][1][0],offset=0,ca_only = True)
Nup133_mol2.add_representation(Nup133_atomic2,resolutions=[beadsize_0,beadsize_3], color=rcolor)
Nup133_mol2.add_representation(Nup133_mol2[:]-Nup133_atomic2,resolutions=[beadsize_4], color=bcolor)
ydimer_mols.append(Nup133_mol2)


# --------------------------------------------------------------
# CREATING CLONES
# --------------------------------------------------------------

#Total 14 mols, 7 for basket and 7X2 for ycomplex
all_spoke_mol = bskt_mols[:]+ydimer_mols[:]
num_bskt_mols = len(bskt_mols)

NPC_MOLS = []

num_chains = len(chain_list['A'])

from collections import OrderedDict
clone_mols=OrderedDict()
for mol in all_spoke_mol:
	mol_hier = mol.get_hierarchy()
	mol_chain_id = IMP.atom.get_chain_id(mol_hier)
	clone_mols[mol]=[mol]

for nc in range(num_chains):

	for mol in all_spoke_mol:
		mol_hier = mol.get_hierarchy()
		mol_chain_id = IMP.atom.get_chain_id(mol_hier)

		#All clones of a particular protein should go into it
		newchain = chain_list[mol_chain_id][nc]
		clone = mol.create_clone(newchain)
		clone_mols[mol].append(clone)
		
for k,v in clone_mols.items():
	NPC_MOLS.append(v)


hier = s.build()
#IMP.atom.show_with_representations(hier)

# --------------------------------------------------------------
# Setup degrees of freedom
# --------------------------------------------------------------

dof = IMP.pmi.dof.DegreesOfFreedom(mdl)


# --------------------------------------------------------------
#  RIGID BODIES from MLPs Coiled-Coil segments
# --------------------------------------------------------------

#Iterate over self and clones copy pairs of mlp1-mlp2

mlp1_added_rgbs=[];
mlp2_added_rgbs=[];


for nc, (rtmol1,rtmol2) in enumerate(zip(NPC_MOLS[0],NPC_MOLS[1])):
	
	for idx, ccseg in enumerate(cc_segments):
		
		mol1_rgb_sel_LR =IMP.atom.Selection(hier,
			molecule="mlp",
			resolution=IMP.atom.ALL_RESOLUTIONS,
			copy_index = 2*nc,
			residue_indexes=range(ccseg[0][0],ccseg[0][1]+1))

		mol2_rgb_sel_LR =IMP.atom.Selection(hier,
			molecule="mlp",
			resolution=IMP.atom.ALL_RESOLUTIONS,
			copy_index = 2*nc+1,
			residue_indexes=range(ccseg[1][0],ccseg[1][1]+1))

		mol1_rgb_sel_LR_temp = []
		for rgbp in mol1_rgb_sel_LR.get_selected_particles():
			if rgbp in mlp1_added_rgbs:
				continue
			mol1_rgb_sel_LR_temp.append(rgbp)
			mlp1_added_rgbs.append(rgbp)

		mol2_rgb_sel_LR_temp = []
		for rgbp in mol2_rgb_sel_LR.get_selected_particles():
			if rgbp in mlp2_added_rgbs:
				continue
			mol2_rgb_sel_LR_temp.append(rgbp)
			mlp2_added_rgbs.append(rgbp)

		mol1_rgb_sel_densities =IMP.atom.Selection(hier,
			molecule="mlp",
			copy_index = 2*nc,
			residue_indexes=range(ccseg[0][0],ccseg[0][1]+1),
			representation_type=IMP.atom.DENSITIES)

		mol2_rgb_sel_densities =IMP.atom.Selection(hier,
			molecule="mlp",
			copy_index = 2*nc+1,
			residue_indexes=range(ccseg[1][0],ccseg[1][1]+1),
			representation_type=IMP.atom.DENSITIES)
		
		mol1_rgb_sel_LR_temp_dens = []
		for rgbp in mol1_rgb_sel_densities.get_selected_particles():
			if rgbp in mlp1_added_rgbs:
				continue
			mol1_rgb_sel_LR_temp_dens.append(rgbp)
			mlp1_added_rgbs.append(rgbp)

		mol2_rgb_sel_LR_temp_dens = []
		for rgbp in mol2_rgb_sel_densities.get_selected_particles():
			if rgbp in mlp2_added_rgbs:
				continue
			mol2_rgb_sel_LR_temp_dens.append(rgbp)
			mlp2_added_rgbs.append(rgbp)
		
		mol1_mol2_seg = mol1_rgb_sel_LR_temp + mol2_rgb_sel_LR_temp + mol1_rgb_sel_LR_temp_dens + mol2_rgb_sel_LR_temp_dens
		
		dof.create_rigid_body(rigid_parts=mol1_mol2_seg, max_trans=MC_TRANS)


# --------------------------------------------------------------
#  FLEXIBLE BEADS for Unstructured Regions for MLPs Proteins molecules
# --------------------------------------------------------------

for nc, (rtmol1,rtmol2) in enumerate(zip(NPC_MOLS[0],NPC_MOLS[1])):

	dof.create_flexible_beads(flex_parts=rtmol1.get_non_atomic_residues(),
		max_trans=MC_TRANS)
	dof.create_flexible_beads(flex_parts=rtmol2.get_non_atomic_residues(),
		max_trans=MC_TRANS)


# --------------------------------------------------------------
#  RIGID BODIES from Nup1, Nup2, Nup60
# --------------------------------------------------------------
#NPC_MOLS = [mlp1, mlp2, nup1, nup2.0, nup2.1, nup60.0, nup60.1, Ydimer_mols]

for nc, rtmol0 in enumerate(NPC_MOLS[2]):

	for nup1rgb in nup1_rigid_bodies:

		rtmol0_rgb_sel =IMP.atom.Selection(hier,
				molecule="nup1",
				resolution=IMP.atom.ALL_RESOLUTIONS,
				#resolution=beadsize_3,
				copy_index = nc,
				residue_indexes=range(nup1rgb[0],nup1rgb[1]+1)
				).get_selected_particles()

		dof.create_rigid_body(rigid_parts=rtmol0_rgb_sel,max_trans=MC_TRANS)

for nc, (rtmol1,rtmol2) in enumerate(zip(NPC_MOLS[3],NPC_MOLS[4])):

	#Code for two sepearte rigid bodies for nup2. Not excluding it
	for nup2rgb in nup2_rigid_bodies:
	
		rtmol1_rgb_sel =IMP.atom.Selection(hier,
			molecule="nup2",
			resolution=IMP.atom.ALL_RESOLUTIONS,
			copy_index = 2*nc,
			residue_indexes=range(nup2rgb[0],nup2rgb[1]+1)).get_selected_particles()

		dof.create_rigid_body(rigid_parts=rtmol1_rgb_sel, max_trans=MC_TRANS)

		rtmol2_rgb_sel =IMP.atom.Selection(hier,
			molecule="nup2",
			resolution=IMP.atom.ALL_RESOLUTIONS,
			copy_index = 2*nc+1,
			residue_indexes=range(nup2rgb[0],nup2rgb[1]+1)).get_selected_particles()

		dof.create_rigid_body(rigid_parts=rtmol2_rgb_sel, max_trans=MC_TRANS)

	#Create super rigid body parts for the nup2 copies since they are connected via FG repeats
	dof.create_super_rigid_body(srb_parts=rtmol1.get_residues(),max_trans=MC_TRANS)
	dof.create_super_rigid_body(srb_parts=rtmol2.get_residues(),max_trans=MC_TRANS)


for nc, (rtmol3,rtmol4) in enumerate(zip(NPC_MOLS[5],NPC_MOLS[6])):

	for nup60rgb in nup60_rigid_bodies:
	
		rtmol1_rgb_sel =IMP.atom.Selection(hier,
			molecule="nup60",
			resolution=IMP.atom.ALL_RESOLUTIONS,
			copy_index = 2*nc,
			residue_indexes=range(nup60rgb[0],nup60rgb[1]+1)).get_selected_particles()

		dof.create_rigid_body(rigid_parts=rtmol1_rgb_sel, max_trans=MC_TRANS)

		rtmol2_rgb_sel =IMP.atom.Selection(hier,
			molecule="nup60",
			resolution=IMP.atom.ALL_RESOLUTIONS,
			copy_index = 2*nc + 1,
			residue_indexes=range(nup60rgb[0],nup60rgb[1]+1)).get_selected_particles()

		dof.create_rigid_body(rigid_parts=rtmol2_rgb_sel, max_trans=MC_TRANS)


# --------------------------------------------------------------
#  FLEXIBLE BEADS for Unstructured Regions for All Proteins molecules
# --------------------------------------------------------------

#NPC_MOLS = [mlp1, mlp2, nup1, nup2.0, nup2.1, nup60.0, nup60.1, Ydimer_mols]
for non_mlptmol in NPC_MOLS[2:num_bskt_mols]:
	
	#Iterate over all clones
	for mol in non_mlptmol:

		dof.create_flexible_beads(flex_parts=mol.get_non_atomic_residues(),
			max_trans=MC_TRANS)



# --------------------------------------------------------------
# YDIMER  Rigid Bodies and flexible beads
# --------------------------------------------------------------

#Get rigid (structured) and flexible parts (unstructured) for Ydimer models 
#NPC_MOLS = [mlp1, mlp2, nup1, nup2.0, nup2.1, nup60.0, nup60.1, Ydimer_mols]
for ydimermol in NPC_MOLS[num_bskt_mols:]:
	
	for mol in ydimermol:

		dof.create_rigid_body(
			rigid_parts=mol.get_atomic_residues(),
			max_trans=MC_TRANS)

		dof.create_flexible_beads(
			flex_parts=mol.get_non_atomic_residues(),
			max_trans=MC_TRANS)


# --------------------------------------------------------------
# TShuffle Spoke Coordinates to Approximate Bounding Box
# --------------------------------------------------------------

zaxis_offset = 150
xaxis_offset = 0;
yaxis_offset = -150;
gmm_com = [-25.1919+xaxis_offset, 436.434+yaxis_offset, -623.066+zaxis_offset]

BBox_Size=600
BBx1,BBy1,BBz1 = gmm_com[0]-BBox_Size/2, gmm_com[1]-BBox_Size/2, gmm_com[2]-BBox_Size/2
BBx2,BBy2,BBz2 = gmm_com[0]+BBox_Size/2, gmm_com[1]+BBox_Size/2, gmm_com[2]+BBox_Size/2

# Shuffle only basket particles, exclude ydimer
IMP.pmi.tools.shuffle_configuration(NPC_MOLS[0:num_bskt_mols],
	bounding_box = ((BBx1,BBy1,BBz1),(BBx2,BBy2,BBz2)),
	max_translation=INIT_TRANS,
	)


# --------------------------------------------------------------
# Setup Symmetry Constraint
# --------------------------------------------------------------

center = IMP.algebra.Vector3D([0, 0, 0])

#Add constraint to only one part of the rigid body NOT both. Mlp1(first mol) was chosen
for rtmol in [NPC_MOLS[0]]:
	for nc in range(num_chains): #nc will be 0 and 1 for three spokes

		rot = IMP.algebra.get_rotation_about_axis([0, 0, 1], 2*math.pi*(-2*nc+1)/8)
		transform = IMP.algebra.get_rotation_about_point(center, rot)

		print("chains:",nc)
		dof.constrain_symmetry(
			rtmol[0].get_atomic_residues(),
			rtmol[nc+1].get_atomic_residues(),
			transform,
			resolution=IMP.atom.ALL_RESOLUTIONS)


#Add constraint for rigid bodies of all other proteins. 
for rtmol in NPC_MOLS[2:]:
	for nc in range(num_chains):
		rot = IMP.algebra.get_rotation_about_axis([0, 0, 1], 2*math.pi*(-2*nc+1)/8)
		transform = IMP.algebra.get_rotation_about_point(center, rot)

		dof.constrain_symmetry(
			rtmol[0].get_atomic_residues(),
			rtmol[nc+1].get_atomic_residues(), 
			transform,
			resolution=IMP.atom.ALL_RESOLUTIONS)

#Add constraint for flexible beads of all proteins
for rtmol in NPC_MOLS:
	for nc in range(num_chains):
		rot = IMP.algebra.get_rotation_about_axis([0, 0, 1], 2*math.pi*(-2*nc+1)/8)
		transform = IMP.algebra.get_rotation_about_point(center, rot)
		
		dof.constrain_symmetry(
			rtmol[0].get_non_atomic_residues(),
			rtmol[nc+1].get_non_atomic_residues(), 
			transform,
			resolution=IMP.atom.ALL_RESOLUTIONS)

mdl.update()  # propagates coordinates

# write a single-frame RMF to view the model
out = IMP.pmi.output.Output()
out.init_rmf(out_path+"/post_shuffle.rmf3", hierarchies=[hier])
out.write_rmf(out_path+"/post_shuffle.rmf3")
out.close_rmf(out_path+"/post_shuffle.rmf3")

#################################################################

#NPC_MOLS = [mlp1, mlp2, nup1, nup2.0, nup2.1, nup60.0, nup60.1, Ydimer_mols]
#All Spoke 1 mols (main spoke)
SPOKE_1_MOLS = []
for npcmols in NPC_MOLS:
	SPOKE_1_MOLS.append(npcmols[0])

#All spoke 1 basket proteins (mlp1, mlp2, nup1, nup2.0, nup2.1, nup60.0, nup60.1)
SPOKE_1_MOLS_BSKT = []
for npcmols in NPC_MOLS[0:num_bskt_mols]:
	SPOKE_1_MOLS_BSKT.append(npcmols[0])

#All spoke 1 mlp proteins (mlp1, mlp2)
SPOKE_1_MOLS_MLPS = []
for mlpmols in NPC_MOLS[0:2]:
	SPOKE_1_MOLS_MLPS.append(mlpmols[0])

#All spoke 1 Ydimer proteins
SPOKE_1_MOLS_YDIM = []
for npcmols in NPC_MOLS[num_bskt_mols:]:
	SPOKE_1_MOLS_YDIM.append(npcmols[0])

#All spoke 1 Ydimer proteins, rigid segments
SPOKE_1_MOLS_YDIM_RIGD = []
for npcmols in NPC_MOLS[num_bskt_mols:]:
	SPOKE_1_MOLS_YDIM_RIGD.append(npcmols[0].get_atomic_residues())

#All spoke 1 Ydimer proteins, flexible beads
SPOKE_1_MOLS_YDIM_FLEX = []
for npcmols in NPC_MOLS[num_bskt_mols:]:
	SPOKE_1_MOLS_YDIM_FLEX.append(npcmols[0].get_non_atomic_residues())


# --------------------------------------------------------------
#  CONNECTIVITY RESTRAINTS FOR FIRST SPOKE ONLY EXCLUDING CLONES
# --------------------------------------------------------------

output_objects = []  # keep a list of functions that need to be reported
for mol in SPOKE_1_MOLS:
	cr = IMP.pmi.restraints.stereochemistry.ConnectivityRestraint(mol)
	cr.add_to_model()
	output_objects.append(cr)



# --------------------------------------------------------------
#  DISTANCE RESTRAINT BETWEEN DOMAINS AND DOMAINS SEPERATED BY FG REPEATS
# --------------------------------------------------------------

n60res1 = int((nup60_n2bm_domain[0]+nup60_n2bm_domain[1])/2)
n2res1 =  int((nup2_n60bm_domain[0]+nup2_n60bm_domain[1])/2)

rppr1 = IMP.pmi.restraints.basic.DistanceRestraint(hier,
		tuple_selection1=(n60res1, n60res1, "nup60",0),
	    tuple_selection2=(n2res1, n2res1, "nup2",0),
		resolution=beadsize_3,
		distancemin = 6.0,
	    distancemax = int(n60bm_n2bm_upperdist),
		label='rppr_nup60_nup2')
rppr1.add_to_model()
output_objects.append(rppr1)

rppr2 = IMP.pmi.restraints.basic.DistanceRestraint(hier,
		tuple_selection1=(n60res1, n60res1, "nup60",1),
	    tuple_selection2=(n2res1, n2res1, "nup2",1),
		resolution=beadsize_3,
		distancemin = 6.0,
	    distancemax = int(n60bm_n2bm_upperdist),
		label='rppr_nup60_nup2')
rppr2.add_to_model()
output_objects.append(rppr2)

rppr3 = IMP.pmi.restraints.basic.DistanceRestraint(hier,
		tuple_selection1=(n60res1, n60res1, "nup60",0),
	    tuple_selection2=(n2res1, n2res1, "nup2",1),
		resolution=beadsize_3,
		distancemin = 6.0,
	    distancemax = int(n60bm_n2bm_upperdist),
		label='rppr_nup60_nup2')
rppr3.add_to_model()
output_objects.append(rppr3)

rppr4 = IMP.pmi.restraints.basic.DistanceRestraint(hier,
		tuple_selection1=(n60res1, n60res1, "nup60",1),
	    tuple_selection2=(n2res1, n2res1, "nup2",0),
		resolution=beadsize_3,
		distancemin = 6.0,
	    distancemax = int(n60bm_n2bm_upperdist),
		label='rppr_nup60_nup2')
rppr4.add_to_model()
output_objects.append(rppr4)

drpr1 = IMP.pmi.restraints.basic.DistanceRestraint(hier,
	    tuple_selection1=(nup2_rigid_bodies[0][1], nup2_rigid_bodies[0][1], "nup2",0),
	    tuple_selection2=(nup2_rigid_bodies[1][0], nup2_rigid_bodies[1][0], "nup2",0),
	    distancemin = 6.0,
	    distancemax = int(nup2_upper_dist),
		resolution=beadsize_3,
		label='drpr1_nup2_nup2_0')
drpr1.add_to_model()
output_objects.append(drpr1)

drpr2 = IMP.pmi.restraints.basic.DistanceRestraint(hier,
	    tuple_selection1=(nup2_rigid_bodies[0][1], nup2_rigid_bodies[0][1], "nup2",1),
	    tuple_selection2=(nup2_rigid_bodies[1][0], nup2_rigid_bodies[1][0], "nup2",1),
	    distancemin = 6.0,
	    distancemax = int(nup2_upper_dist),
		resolution=beadsize_3,
		label='drpr2_nup2_nup2_1')
drpr2.add_to_model()
output_objects.append(drpr2)


drpr3 = IMP.pmi.restraints.basic.DistanceRestraint(hier,
	    tuple_selection1=(nup60_FGRepeat[0][0]-1, nup60_FGRepeat[0][0]-1, "nup60",0),
	    tuple_selection2=(nup60_FGRepeat[0][1]+1, nup60_FGRepeat[0][1]+1, "nup60",0),
	    distancemin = 6.0,
	    distancemax = int(nup60_upper_dist),
		resolution=beadsize_3,
		label='drpr1_nup60_nup60_0')
drpr3.add_to_model()
output_objects.append(drpr3)

drpr4 = IMP.pmi.restraints.basic.DistanceRestraint(hier,
	    tuple_selection1=(nup60_FGRepeat[0][0]-1, nup60_FGRepeat[0][0]-1, "nup60",1),
	    tuple_selection2=(nup60_FGRepeat[0][1]+1, nup60_FGRepeat[0][1]+1, "nup60",1),
	    distancemin = 6.0,
	    distancemax = int(nup60_upper_dist),
		resolution=beadsize_3,
		label='drpr2_nup60_nup60_1')
drpr4.add_to_model()
output_objects.append(drpr4)


# ---------------------------------------------------------
#Disable all the movers from ydimer and its clones as it is a fixed rigid body
for ydimermols in NPC_MOLS[num_bskt_mols:]:
	for mol in ydimermols:
		dof.disable_movers(mol.get_atomic_residues())



# --------------------------------------------------------------
#  Z AXIAL RESTRAINTS
# --------------------------------------------------------------

if include_zaxial_position_restraint==True:

	#Lower bound: Min(Z cord of Y ring)
	#Upper bound: Mean(Z cord of Y ring)
	zapr_0 = IMP.pmi.restraints.npc.ZAxialPositionRestraint(
		hier,protein=(mlp1_nbs_domain_loop[0],mlp1_nbs_domain_loop[1],'mlp',0),
		lower_bound=-280,upper_bound=-230,
		weight=1.0)

	zapr_0.add_to_model()
	output_objects.append(zapr_0)

	zapr_0_1 = IMP.pmi.restraints.npc.ZAxialPositionRestraint(
		hier,protein=(mlp1_nbs_domain_loop[0],mlp1_nbs_domain_loop[1],'mlp',1),
		lower_bound=-280,upper_bound=-230,
		weight=1.0)

	zapr_0_1.add_to_model()
	output_objects.append(zapr_0_1)

	#mlp mol 0 term='C', 10 residues
	zapr_1 = IMP.pmi.restraints.npc.ZAxialPositionRestraint(
			hier,protein=(mlp1_cterm[1]-10,mlp1_cterm[1],'mlp',0),
			lower_bound=-700,upper_bound=-600,
			weight=1.0)
	zapr_1.add_to_model()
	output_objects.append(zapr_1)

	#mlp mol 0 term='N' 10 residues
	zapr_2 = IMP.pmi.restraints.npc.ZAxialPositionRestraint(
			hier,protein=(mlp1_nterm[0],mlp1_nterm[0]+10,'mlp',0),
			lower_bound=-700,upper_bound=-550,
			weight=1.0)
	zapr_2.add_to_model()
	output_objects.append(zapr_2)

	#mlp mol 1 term='C', 10 residues
	zapr_3 = IMP.pmi.restraints.npc.ZAxialPositionRestraint(
			hier,protein=(mlp1_cterm[1]-10,mlp1_cterm[1],'mlp',1),
			lower_bound=-700,upper_bound=-600,
			weight=1.0)
	zapr_3.add_to_model()
	output_objects.append(zapr_3)

	#mlp mol 1 term='N' 10 residues
	zapr_4 = IMP.pmi.restraints.npc.ZAxialPositionRestraint(
			hier,protein=(mlp1_nterm[0],mlp1_nterm[0]+10,'mlp',1),
			lower_bound=-700,upper_bound=-550,
			weight=1.0)
	zapr_4.add_to_model()
	output_objects.append(zapr_4)
	


# --------------------------------------------------------------
#  MEMBRANE RESTRAINTS
# --------------------------------------------------------------

tor_th      = 45.0
tor_th_ALPS = 12.0
tor_R       = 390.0 + 150.0
tor_r       = 150.0 - tor_th/2.0
tor_r_ALPS  = 150.0 - tor_th_ALPS/2.0
msl_sigma   = 1.0
#msl_weight  = MSLR_WEIGHT
if include_membrane_surface_binding:

	membrane_surface_sel = [
							(nup1_helix_res[0][0],nup1_helix_res[0][1],'nup1',0),
							(nup60_helix_res[0][0],nup60_helix_res[0][1],'nup60',0),
							(nup60_helix_res[0][0],nup60_helix_res[0][1],'nup60',1)]
	
	for sel in membrane_surface_sel:
		print('Applying membrane localization restraint:', sel)

		msl_1 = IMP.pmi.restraints.npc.MembraneSurfaceLocationRestraint(
			hier,
			protein=sel,
			tor_R=tor_R,
			tor_r=tor_r,
			tor_th=tor_th_ALPS,
			sigma=msl_sigma,
			resolution = IMP.atom.ALL_RESOLUTIONS)

		msl_1.set_weight(MSLR_WEIGHT)
		msl_1.add_to_model()
		output_objects.append(msl_1)
		print('Membrane binding restraint ready', msl_1.get_output())


# --------------------------------------------------------------
#  CROSSLINK RESTRAINTS FOR ALL MOLECULES INCLUDING CLONES
# --------------------------------------------------------------

if include_crosslink_restraint1:

	cldbkc = IMP.pmi.io.crosslink.CrossLinkDataBaseKeywordsConverter()
	cldbkc.set_protein1_key("Protein1")
	cldbkc.set_protein2_key("Protein2")
	cldbkc.set_residue1_key("Residue1")
	cldbkc.set_residue2_key("Residue2")
	cldbkc.set_unique_id_key("UniqueID")

	cldb = IMP.pmi.io.crosslink.CrossLinkDataBase(cldbkc)
	cldb.create_set_from_file(xlinkfilename_newDSS)
	
	xlr = IMP.pmi.restraints.crosslinking.CrossLinkingMassSpectrometryRestraint(
		root_hier=hier, database=cldb, length=XL_CUT, slope=XL_SLOPE,
		resolution=1.0, label="XLDSS", linker=ihm.cross_linkers.dss, weight=XL_WEIGHT)

	xlr.add_to_model()
	output_objects.append(xlr)

	# needed to sample the nuisance particles (noise params)
	dof.get_nuisances_from_restraint(xlr)
	print("Eval:",xlr.rs.unprotected_evaluate(None))

if include_crosslink_restraint2:

	cldbkc1 = IMP.pmi.io.crosslink.CrossLinkDataBaseKeywordsConverter()
	cldbkc1.set_protein1_key("Protein1")
	cldbkc1.set_protein2_key("Protein2")
	cldbkc1.set_residue1_key("Residue1")
	cldbkc1.set_residue2_key("Residue2")
	cldbkc1.set_unique_id_key("UniqueID")

	cldb1 = IMP.pmi.io.crosslink.CrossLinkDataBase(cldbkc1)
	cldb1.create_set_from_file(xlinkfilename_other)
	
	xlr1 = IMP.pmi.restraints.crosslinking.CrossLinkingMassSpectrometryRestraint(
		root_hier=hier, database=cldb1, length=XL_CUT, slope=XL_SLOPE,
		resolution=1.0, label="XL", linker=ihm.cross_linkers.dss, weight=XL_WEIGHT1)

	xlr1.add_to_model()
	output_objects.append(xlr1)

	# needed to sample the nuisance particles (noise params)
	dof.get_nuisances_from_restraint(xlr1)
	print("Eval:",xlr1.rs.unprotected_evaluate(None))



# --------------------------------------------------------------
#  SAMPLING
# --------------------------------------------------------------

# Quickly move all flexible beads into place
dof.optimize_flexible_beads(optimize_flex_beads_steps)

# write a single-frame RMF to view the model
out = IMP.pmi.output.Output()
out.init_rmf(out_path+"/REX1_initial_model.rmf3", hierarchies=[hier])
out.write_rmf(out_path+"/REX1_initial_model.rmf3")
out.close_rmf(out_path+"/REX1_initial_model.rmf3")

# Run replica exchange Monte Carlo sampling
rex1 = IMP.pmi.macros.ReplicaExchange(
	mdl,
	root_hier=hier,
	monte_carlo_sample_objects=dof.get_movers(),
	global_output_directory=out_path+'/REX1/',
	output_objects=output_objects,
	write_initial_rmf=True,
	monte_carlo_steps=MC_STEPS_PER_FRAME,
	number_of_best_scoring_models=0,
	number_of_frames=NUM_FRAMES_1,
	score_moved=True,
	replica_exchange_minimum_temperature=MIN_REX_TEMP,
	replica_exchange_maximum_temperature=MAX_REX_TEMP)

# time this step
t0 = time.time()
rex1.execute_macro()
dt1 = time.time() - t0
print("Time taken for REX1:",dt1)


# --------------------------------------------------------------
#  GAUSSIAN EMRESTRAINTS FOR FIRST BASKET PROTEINS ONLY EXCLUDING YDIMER and CLONES
# --------------------------------------------------------------
if include_gaussian_em_restraint:

	mlp1_densities = IMP.atom.Selection(hier,molecule="mlp",copy_index = 0,representation_type=IMP.atom.DENSITIES).get_selected_particles()

	mlp2_densities = IMP.atom.Selection(hier,molecule="mlp",copy_index = 1,representation_type=IMP.atom.DENSITIES).get_selected_particles()

	densities = mlp1_densities + mlp2_densities

	gem = IMP.bayesianem.restraint.GaussianEMRestraintWrapper(densities, gmmfilename,
		scale_target_to_mass=True,slope=EM_SLOPE)
	gem.set_label("EM")
	gem.add_to_model()
	gem.set_weight(weight=EMR_WEIGHT)
	output_objects.append(gem)

	t0 = gem.evaluate()
	mdl.update()


# --------------------------------------------------------------
#  EXCLUDED VOLUME RESTRAINTS
# --------------------------------------------------------------

#Excluded volume for main spoke
evr0 = IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere(
	included_objects=SPOKE_1_MOLS,
	resolution=EV_RESOLUTION)
evr0.add_to_model()
output_objects.append(evr0)


# write a single-frame RMF to view the model
out = IMP.pmi.output.Output()
out.init_rmf(out_path+"/REX2_initial_model.rmf3", hierarchies=[hier])
out.write_rmf(out_path+"/REX2_initial_model.rmf3")
out.close_rmf(out_path+"/REX2_initial_model.rmf3")

rex1_ob = rex1.get_replica_exchange_object()

# Run replica exchange Monte Carlo sampling
rex2 = IMP.pmi.macros.ReplicaExchange(
	mdl,
	root_hier=hier,
	monte_carlo_sample_objects=dof.get_movers(),
	global_output_directory=out_path+'/REX2/',
	output_objects=output_objects,
	write_initial_rmf=True,
	monte_carlo_steps=MC_STEPS_PER_FRAME,
	number_of_best_scoring_models=0,
	number_of_frames=NUM_FRAMES_2,
	replica_exchange_maximum_temperature=MAX_REX_TEMP,
	score_moved=True,
	replica_exchange_object = rex1_ob)

# time this step
t1 = time.time()
rex2.execute_macro()
dt2 = time.time() - t1
print("Time taken for REX2:",dt2)

