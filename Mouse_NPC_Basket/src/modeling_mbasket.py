#!/usr/bin/env python

#####################################################
# Last Update: March 26, 2024 by Neelesh Soni
# Andrej Sali group, University of California San Francisco (UCSF)
#####################################################

# --------------------------------------------------------------
# Import IMP modules
# --------------------------------------------------------------
import IMP
import RMF
import IMP.atom
import IMP.core
import IMP.rmf
import IMP.pmi
import IMP.pmi.topology
import IMP.pmi.dof
import IMP.pmi.macros
import IMP.pmi.io
import IMP.pmi.io.crosslink
import ihm.cross_linkers
from IMP.pmi.io.crosslink import CrossLinkDataBaseKeywordsConverter
import IMP.pmi.restraints
import IMP.bayesianem
import IMP.bayesianem.restraint
import IMP.pmi.restraints.basic
import IMP.pmi.restraints.stereochemistry
import IMP.pmi.restraints.crosslinking
import IMP.pmi.restraints.npc
from IMP.pmi.io.crosslink import FilterOperator as FO

# --------------------------------------------------------------
# Import other python modules
# --------------------------------------------------------------

import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import operator
import random
import numpy as np
import glob
import sys
import os
from collections import defaultdict, OrderedDict
from argparse import ArgumentParser, RawDescriptionHelpFormatter


# -----------------------------------------
# Coiled-COil Segment Definitions
# -----------------------------------------

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
		
		respair1 = list(map(int,toks[1].split('-')))
		respair2 = list(map(int,toks[3].split('-')))

		cc_segments.append([ respair1, respair2 ])

	return cc_segments

def get_loop_regions(cc_segments, nterm_prot1, cterm_prot1, nterm_prot2, cterm_prot2):
	"""Summary
	
	Args:
		cc_segments (TYPE): Coiled-coil segments
		nterm_prot1 (TYPE): N-terminus of prot1
		cterm_prot1 (TYPE): C-terminus of prot1
		nterm_prot2 (TYPE): N-terminus of prot2
		cterm_prot2 (TYPE): C-terminus of prot2
	
	Returns:
		TYPE: Return loop segemnts
	"""
	loop_segments = []

	i = 0
	ccseg_curr = cc_segments[i]

	loop_segments.append([ [ nterm_prot1, ccseg_curr[0][0]-1 ], [ nterm_prot2, ccseg_curr[1][0]-1 ] ])
	
	for i in range(0,len(list(cc_segments))-1):

		ccseg_curr = cc_segments[i]
		ccseg_next = cc_segments[i+1]

		loopx_prot1 = [ccseg_curr[0][1]+1, ccseg_next[0][0]-1]
		loopx_prot2 = [ccseg_curr[1][1]+1, ccseg_next[1][0]-1]

		if ((loopx_prot1[0] <= loopx_prot1[1]) and (loopx_prot2[0] <= loopx_prot2[1]) ):
			loop_segments.append([loopx_prot1, loopx_prot2])

	i = len(list(cc_segments))-1
	ccseg_curr = cc_segments[i]
	loop_segments.append([ [ ccseg_curr[0][1]+1, cterm_prot1 ], [ ccseg_curr[1][1]+1, cterm_prot2 ] ])


	return loop_segments

# -----------------------------------------
# Command Line arguments Definitions
# -----------------------------------------

# Parse commandline Inputs
parser_obj = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
parser_obj.add_argument("-o", "--output_dir", default="output", help="Output files directory")
parser_obj.add_argument("-d", "--data_dir", default="data", help="Data files directory")

args = parser_obj.parse_args()
data_path = os.path.abspath(args.data_dir)
out_path = os.path.abspath(args.output_dir)

include_Occams = True
include_EM_spec2 = True
include_zaxial_position_restraint=True
include_membrane_surface_binding=True

print("Following Command Line Argumenst were used: \n")
print("data path:",data_path)
print("output path:",out_path)


Prots_Rigid_Bodies = OrderedDict()

mlps_pdb_fn=os.path.join(data_path,'pdbs/Model_A_run52_clus13_highres_C8_7.pdb')

nup50_pdb_fn=os.path.join(data_path,'pdbs/Nup50_loopremoved.pdb')
nup153_pdb_fn=os.path.join(data_path,'pdbs/Nup153.pdb')

alifname = os.path.join(data_path,'alignments/MLP_MLP_alignment.cali')
mlp1_fastafile = os.path.join(data_path,'fasta/MLP.fasta')
mlp2_fastafile = os.path.join(data_path,'fasta/MLP.fasta')

nup2_fastafile = os.path.join(data_path,'fasta/Nup2.fasta') #Homolog of Mouse Nup50 
nup60_fastafile = os.path.join(data_path,'fasta/Nup60.fasta') #Homolog of Mouse Nup153 

tprs_pdb_fn=os.path.join(data_path,'pdbs/Model_A_MuTPR_aligned.pdb')
tpr_alifname = os.path.join(data_path,'alignments/MuTPR_MuTPR_alignment.cali')
tpr1_fastafile = os.path.join(data_path,'fasta/MuTPR.fasta')
tpr2_fastafile = os.path.join(data_path,'fasta/MuTPR.fasta')

nup50_fastafile = os.path.join(data_path,'fasta/Nup50.fasta')  #Homolog of Yeast Nup2
nup153_fastafile = os.path.join(data_path,'fasta/Nup153.fasta') #Homolog of Yeast Nup60

gmmfilename = os.path.join(data_path,'gmms/1_subunits_Mammalian_bskt_noNR_center_ng100.txt')

ydimer_fastafile = os.path.join(data_path,'fasta/Y-dimer.fasta')
Ydimer_seqs = IMP.pmi.topology.Sequences(ydimer_fastafile)
ydimer = os.path.join(data_path,'pdbs/Model_A_run52_clus13_highres_C8_7.pdb')

YDIMER={'ytNup120':['A','L'],'ytNup85':['B','M'],'ytNup145c':['C','N'],
		'ytSec13':['D','O'],'ytSeh1':['E','P'],'ytNup84':['F','Q'],
		'ytNup133':['G','R']}

muydimer_fastafile = os.path.join(data_path,'fasta/muY-dimer.fasta')
muYdimer_seqs = IMP.pmi.topology.Sequences(muydimer_fastafile)
muydimer = os.path.join(data_path,'pdbs/Mouse_Ycom_Aligned_TomoMap_Renamed.pdb')


muYDIMER={'muNup160':['A','L'],'muNup85':['B','M'],'muNup96':['C','N'],
		'muSec13':['D','O'],'muSeh1':['E','P'],'muNup107':['F','Q'],
		'muNup133':['G','R'],'muNup43':['J','X'],'muNup37':['K','Y']}

EV_RESOLUTION=30
INIT_TRANS = 500.0

# rigid body/beads params
beadsize_0 = 1
beadsize_1 = 5 #For high resoluton particles. 2 reduces the beads to half. 3/4/5 do not change too much
beadsize_2 = 30 #For medium resolution particles
beadsize_3 = 30 #For medium resolution particles
beadsize_4 = 50 #For low resolution particles
beadsize_5 = 30 #For low resolution particles

EM_SLOPE = 0.2
EMR_WEIGHT = 1.0

MSLR_WEIGHT = 1.0

MC_TRANS = 10.0
MC_ROT   = 0.2 * (22/7)

# replica exchange params
MIN_REX_TEMP = 1.0
MAX_REX_TEMP = 4.0

optimize_flex_beads_steps=10
MC_STEPS_PER_FRAME = 20
NUM_FRAMES_1 = 500
NUM_FRAMES_2 = 300000

#CLONES CHAINS 
#NO CLONES AT ALL
chain_list = {'A':'','a':''}

# -----------------------------------------
# Read Fasta Files
# -----------------------------------------

mlp1_seqs = IMP.pmi.topology.Sequences(mlp1_fastafile)
mlp2_seqs = IMP.pmi.topology.Sequences(mlp2_fastafile)
#nup1_seqs = IMP.pmi.topology.Sequences(nup1_fastafile)
nup2_seqs = IMP.pmi.topology.Sequences(nup2_fastafile)
nup60_seqs = IMP.pmi.topology.Sequences(nup60_fastafile)

tpr1_seqs = IMP.pmi.topology.Sequences(tpr1_fastafile)
tpr2_seqs = IMP.pmi.topology.Sequences(tpr2_fastafile)
nup153_seqs = IMP.pmi.topology.Sequences(nup153_fastafile)
nup50_seqs = IMP.pmi.topology.Sequences(nup50_fastafile)

#Get cc segments from the alignment file
mlp_cc_segments = read_alignment_file(alifname)

mlp1_cterm = len(mlp1_seqs['MLP'+':A']);mlp2_cterm = len(mlp2_seqs['MLP'+':A'])
mlp1_nterm = 1;mlp2_nterm = 1
mlp_loop_segments = get_loop_regions(mlp_cc_segments, mlp1_nterm, mlp1_cterm, mlp2_nterm, mlp2_cterm)

#Get cc segments from the alignment file
tpr_cc_segments = read_alignment_file(tpr_alifname)

tpr1_cterm = len(tpr1_seqs['MuTPR'+':A']);tpr2_cterm = len(tpr1_seqs['MuTPR'+':A'])
tpr1_nterm = 1;tpr2_nterm = 1
tpr_loop_segments = get_loop_regions(tpr_cc_segments, tpr1_nterm, tpr1_cterm, tpr2_nterm, tpr2_cterm)

tpr_mlp_aligned_fastafile = os.path.join(data_path,'alignments/MuTPR_MLP_aligned.fasta')
tpr_mlp_aligned_seqs = IMP.pmi.topology.Sequences(tpr_mlp_aligned_fastafile)

mlp1_nbs_domain_loop = [488,530] #For z-axial restraint for NBS domain
tpr_nbs_domain_loop = [578,608] #This is obtained using MLP TPR alignment

nup50_rigid_bodies = [[151,204],[355,466]]
nup50_nonrigid_residues = [[1,150],[205,354]]

nup153_rigid_bodies = [[36,57]]
nup153_nonrigid_residues = [[1,35],[58,428],[540,574]] #residues come from alignments

nup153_membrane_res = [[36,57]]

# -----------------------------------------
# Define Colors for Proteins
# -----------------------------------------

#Color map iterator in the Chimera format
Rcolor1 = cm.get_cmap('Set3', 12) #No of colors in Set3 = 12
Rcolor2 = cm.get_cmap('Pastel1', 8) #No of colors in Set2 = 8
Rcolor3 = cm.get_cmap('tab20', 20) #No of colors in Set2 = 8

Rcolors = iter(np.vstack(( Rcolor1(np.linspace(0,1,12))[:,0:3], Rcolor2(np.linspace(0,1,8))[:,0:3], Rcolor3(np.linspace(0,1,20))[:,0:3] )) )

ytbskt_mols =[]
ytbskt_nup_mols = []
ytydimer_mols=[]

mubskt_mols =[]
mubskt_nup_mols = []
muydimer_mols=[]

mdl = IMP.Model()
s1 = IMP.pmi.topology.System(mdl)

st1 = s1.create_state()

# --------------------------------------------------------------
# TPR molecule (Copy 0)
# --------------------------------------------------------------

rcolor = next(Rcolors);bcolor = rcolor #next(Bcolors)
tpr1_mol1 = st1.create_molecule('tpr', sequence=tpr1_seqs['MuTPR'+':A'], chain_id='A')
tpr1_atomic1 = tpr1_mol1.add_structure(tprs_pdb_fn, chain_id='H',offset=0, ca_only = True)

#CC DOmains
for i, ccseg in enumerate(list(tpr_cc_segments)):
	tpr1_mol1.add_representation(tpr1_mol1[ccseg[0][0]-1:ccseg[0][1]],
		resolutions=[beadsize_0,beadsize_3], color=rcolor,
		#resolutions=[beadsize_0], color=rcolor,
		density_residues_per_component=20,
		density_prefix=out_path+'/tpr1_gmm'+str(i),
		density_force_compute=False,
		density_voxel_size=10.0)

#CC DOmain Linkers
for i in range(1,len(list(tpr_loop_segments))-1):
	ccseg = list(tpr_loop_segments)[i]
	tpr1_mol1.add_representation(tpr1_mol1[ccseg[0][0]-1:ccseg[0][1]],
		resolutions=[beadsize_2], color=rcolor,
		setup_particles_as_densities=True)

#NTD/CTD
for i in [0,len(list(tpr_loop_segments))-1]:
	ccseg = list(tpr_loop_segments)[i]
	tpr1_mol1.add_representation(tpr1_mol1[ccseg[0][0]-1:ccseg[0][1]],
		resolutions=[beadsize_4], color=rcolor,
		setup_particles_as_densities=True)

mubskt_mols.append(tpr1_mol1)


# --------------------------------------------------------------
# TPR molecule (Copy 1)
# --------------------------------------------------------------

rcolor = next(Rcolors);bcolor = rcolor #next(Bcolors)
tpr2_mol1 = tpr1_mol1.create_copy( chain_id='a')
tpr2_atomic1 = tpr2_mol1.add_structure(tprs_pdb_fn, chain_id='I',offset=0, ca_only = True)

#CC DOmains
for i, ccseg in enumerate(list(tpr_cc_segments)):
	tpr2_mol1.add_representation(tpr2_mol1[ccseg[1][0]-1:ccseg[1][1]],
		resolutions=[beadsize_0,beadsize_3], color=rcolor,
		#resolutions=[beadsize_0], color=rcolor,
		density_residues_per_component=20,
		density_prefix=out_path+'/tpr2_gmm'+str(i),
		density_force_compute=False,
		density_voxel_size=10.0)

#CC DOmain Linkers
for i in range(1,len(list(tpr_loop_segments))-1):
	ccseg = list(tpr_loop_segments)[i]

	tpr2_mol1.add_representation(tpr2_mol1[ccseg[1][0]-1:ccseg[1][1]],
		resolutions=[beadsize_2], color=rcolor,
		setup_particles_as_densities=True)

#NTD/CTD
for i in [0,len(list(tpr_loop_segments))-1]:
	ccseg = list(tpr_loop_segments)[i]

	tpr2_mol1.add_representation(tpr2_mol1[ccseg[1][0]-1:ccseg[1][1]],
		resolutions=[beadsize_4], color=rcolor,
		setup_particles_as_densities=True)

mubskt_mols.append(tpr2_mol1)

# --------------------------------------------------------------
# NUP50 molecule Mouse (copy 0)
# --------------------------------------------------------------

rcolor = next(Rcolors);bcolor = rcolor #next(Bcolors)
Nup50_mol0 = st1.create_molecule('nup50', sequence=nup50_seqs['NUP50'+':A'], chain_id='A')
Nup50_atomic0 = Nup50_mol0.add_structure(tprs_pdb_fn, chain_id='V',offset=0,ca_only = True)

for nup50rgb in nup50_rigid_bodies:
	Nup50_mol0.add_representation(Nup50_mol0[nup50rgb[0]-1:nup50rgb[1]],
		resolutions=[beadsize_0,beadsize_3], color=rcolor)

for nup50loop in nup50_nonrigid_residues:
	Nup50_mol0.add_representation(Nup50_mol0[nup50loop[0]-1:nup50loop[1]],
		resolutions=[beadsize_3], color=rcolor)

mubskt_mols.append(Nup50_mol0)

# --------------------------------------------------------------
# NUP50 molecule Mouse (copy 1)
# --------------------------------------------------------------

rcolor = next(Rcolors);bcolor = rcolor #next(Bcolors)
Nup50_mol1 = Nup50_mol0.create_copy(chain_id='a')
Nup50_atomic1 = Nup50_mol1.add_structure(tprs_pdb_fn, chain_id='W',offset=0,ca_only = True)

for nup50rgb in nup50_rigid_bodies:
	Nup50_mol1.add_representation(Nup50_mol1[nup50rgb[0]-1:nup50rgb[1]],
		resolutions=[beadsize_0,beadsize_3], color=rcolor)

for nup50loop in nup50_nonrigid_residues:
	Nup50_mol1.add_representation(Nup50_mol1[nup50loop[0]-1:nup50loop[1]],
		resolutions=[beadsize_3], color=rcolor)

mubskt_mols.append(Nup50_mol1)


# --------------------------------------------------------------
# NUP153 molecule Mouse (copy 0)
# --------------------------------------------------------------

rcolor = next(Rcolors);bcolor = rcolor #next(Bcolors)
Nup153_mol0 = st1.create_molecule('nup153', sequence=nup153_seqs['NUP153'+':A'], chain_id='A')
Nup153_atomic0 = Nup153_mol0.add_structure(tprs_pdb_fn, chain_id='T',offset=0,ca_only = True)

for nup153rgb in nup153_rigid_bodies:
	Nup153_mol0.add_representation(Nup153_mol0[nup153rgb[0]-1:nup153rgb[1]],
		resolutions=[beadsize_0,beadsize_5], color=rcolor)

for nup153loop in nup153_nonrigid_residues:
	Nup153_mol0.add_representation(Nup153_mol0[nup153loop[0]-1:nup153loop[1]],
		resolutions=[beadsize_5], color=rcolor)

mubskt_mols.append(Nup153_mol0)


# --------------------------------------------------------------
# NUP153 molecule Mouse (copy 1)
# --------------------------------------------------------------

rcolor = next(Rcolors);bcolor = rcolor #next(Bcolors)
Nup153_mol1 = Nup153_mol0.create_copy(chain_id='a')
Nup153_atomic1 = Nup153_mol1.add_structure(tprs_pdb_fn, chain_id='U',offset=0,ca_only = True)

for nup153rgb in nup153_rigid_bodies:
	Nup153_mol1.add_representation(Nup153_mol1[nup153rgb[0]-1:nup153rgb[1]],
		resolutions=[beadsize_0,beadsize_5], color=rcolor)

for nup153loop in nup153_nonrigid_residues:
	Nup153_mol1.add_representation(Nup153_mol1[nup153loop[0]-1:nup153loop[1]],
		resolutions=[beadsize_5], color=rcolor)

mubskt_mols.append(Nup153_mol1)

# --------------------------------------------------------------
# Mouse Y-Dimer complex as rigid body, no representation for mising structures
# --------------------------------------------------------------

rcolor = next(Rcolors);bcolor = rcolor #next(Bcolors)
muNup160_mol1 = st1.create_molecule('munup160', sequence=muYdimer_seqs['muNup160'], chain_id='A')
muNup160_atomic1 = muNup160_mol1.add_structure(muydimer, chain_id=muYDIMER['muNup160'][0],offset=0,ca_only = True)
muNup160_mol1.add_representation(muNup160_atomic1,resolutions=[beadsize_0,beadsize_3], color=rcolor)
muydimer_mols.append(muNup160_mol1)

print("Y_N120 : Str Res:",len(muNup160_atomic1),"\tNon_Str Res:",len(muNup160_mol1[:]-muNup160_atomic1))

muNup160_mol2 = muNup160_mol1.create_copy( chain_id='a')
muNup160_atomic2 = muNup160_mol2.add_structure(muydimer, chain_id=muYDIMER['muNup160'][1],offset=0,ca_only = True)
muNup160_mol2.add_representation(muNup160_atomic2,resolutions=[beadsize_0,beadsize_3], color=rcolor)
muydimer_mols.append(muNup160_mol2)

print("Y_N120 : Str Res:",len(muNup160_atomic2),"\tNon_Str Res:",len(muNup160_mol2[:]-muNup160_atomic2))


rcolor = next(Rcolors);bcolor = rcolor #next(Bcolors)
muNup85_mol1 = st1.create_molecule('munup85', sequence=muYdimer_seqs['muNup85'], chain_id='A')
muNup85_atomic1 = muNup85_mol1.add_structure(muydimer, chain_id=muYDIMER['muNup85'][0],offset=0,ca_only = True)
muNup85_mol1.add_representation(muNup85_atomic1,resolutions=[beadsize_0,beadsize_3], color=rcolor)
muydimer_mols.append(muNup85_mol1)

print("Y_N85  : Str Res:",len(muNup85_atomic1),"\tNon_Str Res:",len(muNup85_mol1[:]-muNup85_atomic1))

muNup85_mol2 = muNup85_mol1.create_copy( chain_id='a')
muNup85_atomic2 = muNup85_mol2.add_structure(muydimer, chain_id=muYDIMER['muNup85'][1],offset=0,ca_only = True)
muNup85_mol2.add_representation(muNup85_atomic2,resolutions=[beadsize_0,beadsize_3], color=rcolor)
muydimer_mols.append(muNup85_mol2)

print("Y_N85  : Str Res:",len(muNup85_atomic2),"\tNon_Str Res:",len(muNup85_mol2[:]-muNup85_atomic2))


rcolor = next(Rcolors);bcolor = rcolor #next(Bcolors)
muNup96_mol1 = st1.create_molecule('munup96', sequence=muYdimer_seqs['muNup96'], chain_id='A')
muNup96_atomic1 = muNup96_mol1.add_structure(muydimer, chain_id=muYDIMER['muNup96'][0],offset=0,ca_only = True)
muNup96_mol1.add_representation(muNup96_atomic1,resolutions=[beadsize_0,beadsize_3], color=rcolor)
muydimer_mols.append(muNup96_mol1)

print("Y_N145 : Str Res:",len(muNup96_atomic1),"\tNon_Str Res:",len(muNup96_mol1[:]-muNup96_atomic1))

muNup96_mol2 = muNup96_mol1.create_copy(chain_id='a')
muNup96_atomic2 = muNup96_mol2.add_structure(muydimer, chain_id=muYDIMER['muNup96'][1],offset=0,ca_only = True)
muNup96_mol2.add_representation(muNup96_atomic2,resolutions=[beadsize_0,beadsize_3], color=rcolor)
muydimer_mols.append(muNup96_mol2)

print("Y_N145 : Str Res:",len(muNup96_atomic2),"\tNon_Str Res:",len(muNup96_mol2[:]-muNup96_atomic2))


rcolor = next(Rcolors);bcolor = rcolor #next(Bcolors)
muSec13_mol1 = st1.create_molecule('musec13', sequence=muYdimer_seqs['muSec13'], chain_id='A')
muSec13_atomic1 = muSec13_mol1.add_structure(muydimer, chain_id=muYDIMER['muSec13'][0],offset=0,ca_only = True)
muSec13_mol1.add_representation(muSec13_atomic1,resolutions=[beadsize_0,beadsize_3], color=rcolor)
muydimer_mols.append(muSec13_mol1)

print("Y_Sc13 : Str Res:",len(muSec13_atomic1),"\tNon_Str Res:",len(muSec13_mol1[:]-muSec13_atomic1))

muSec13_mol2 = muSec13_mol1.create_copy(chain_id='a')
muSec13_atomic2 = muSec13_mol2.add_structure(muydimer, chain_id=muYDIMER['muSec13'][1],offset=0,ca_only = True)
muSec13_mol2.add_representation(muSec13_atomic2,resolutions=[beadsize_0,beadsize_3], color=rcolor)
muydimer_mols.append(muSec13_mol2)

print("Y_Sc13 : Str Res:",len(muSec13_atomic2),"\tNon_Str Res:",len(muSec13_mol2[:]-muSec13_atomic2))

rcolor = next(Rcolors);bcolor = rcolor #next(Bcolors)
muSeh1_mol1 = st1.create_molecule('museh1', sequence=muYdimer_seqs['muSeh1'], chain_id='A')
muSeh1_atomic1 = muSeh1_mol1.add_structure(muydimer, chain_id=muYDIMER['muSeh1'][0],offset=0,ca_only = True)
muSeh1_mol1.add_representation(muSeh1_atomic1,resolutions=[beadsize_0,beadsize_3], color=rcolor)
muydimer_mols.append(muSeh1_mol1)

print("Y_Sh1  : Str Res:",len(muSeh1_atomic1),"\tNon_Str Res:",len(muSeh1_mol1[:]-muSeh1_atomic1))

muSeh1_mol2 = muSeh1_mol1.create_copy(chain_id='a')
muSeh1_atomic2 = muSeh1_mol2.add_structure(muydimer, chain_id=muYDIMER['muSeh1'][1],offset=0,ca_only = True)
muSeh1_mol2.add_representation(muSeh1_atomic2,resolutions=[beadsize_0,beadsize_3], color=rcolor)
muydimer_mols.append(muSeh1_mol2)

print("Y_Sh1  : Str Res:",len(muSeh1_atomic2),"\tNon_Str Res:",len(muSeh1_mol2[:]-muSeh1_atomic2))


rcolor = next(Rcolors);bcolor = rcolor #next(Bcolors)
muNup107_mol1 = st1.create_molecule('munup107', sequence=muYdimer_seqs['muNup107'], chain_id='A')
muNup107_atomic1 = muNup107_mol1.add_structure(muydimer, chain_id=muYDIMER['muNup107'][0],offset=0,ca_only = True)
muNup107_mol1.add_representation(muNup107_atomic1,resolutions=[beadsize_0,beadsize_3], color=rcolor)
muydimer_mols.append(muNup107_mol1)

print("Y_N84  : Str Res:",len(muNup107_atomic1),"\tNon_Str Res:",len(muNup107_mol1[:]-muNup107_atomic1))

muNup107_mol2 = muNup107_mol1.create_copy( chain_id='a')
muNup107_atomic2 = muNup107_mol2.add_structure(muydimer, chain_id=muYDIMER['muNup107'][1],offset=0,ca_only = True)
muNup107_mol2.add_representation(muNup107_atomic2,resolutions=[beadsize_0,beadsize_3], color=rcolor)
muydimer_mols.append(muNup107_mol2)

print("Y_N84  : Str Res:",len(muNup107_atomic2),"\tNon_Str Res:",len(muNup107_mol2[:]-muNup107_atomic2))

rcolor = next(Rcolors);bcolor = rcolor #next(Bcolors)
muNup133_mol1 = st1.create_molecule('munup133', sequence=muYdimer_seqs['muNup133'], chain_id='A')
muNup133_atomic1 = muNup133_mol1.add_structure(muydimer, chain_id=muYDIMER['muNup133'][0],offset=0,ca_only = True)
muNup133_mol1.add_representation(muNup133_atomic1,resolutions=[beadsize_0,beadsize_3], color=rcolor)
muydimer_mols.append(muNup133_mol1)

print("Y_N133 : Str Res:",len(muNup133_atomic1),"\tNon_Str Res:",len(muNup133_mol1[:]-muNup133_atomic1))

muNup133_mol2 = muNup133_mol1.create_copy(chain_id='a')
muNup133_atomic2 = muNup133_mol2.add_structure(muydimer, chain_id=muYDIMER['muNup133'][1],offset=0,ca_only = True)
muNup133_mol2.add_representation(muNup133_atomic2,resolutions=[beadsize_0,beadsize_3], color=rcolor)
muydimer_mols.append(muNup133_mol2)

print("Y_N133 : Str Res:",len(muNup133_atomic2),"\tNon_Str Res:",len(muNup133_mol2[:]-muNup133_atomic2))

rcolor = next(Rcolors);bcolor = rcolor #next(Bcolors)
muNup37_mol1 = st1.create_molecule('munup37', sequence=muYdimer_seqs['muNup37'], chain_id='A')
muNup37_atomic1 = muNup37_mol1.add_structure(muydimer, chain_id=muYDIMER['muNup37'][0],offset=0,ca_only = True)
muNup37_mol1.add_representation(muNup37_atomic1,resolutions=[beadsize_0,beadsize_3], color=rcolor)
muydimer_mols.append(muNup37_mol1)

print("Y_N37  : Str Res:",len(muNup37_atomic1),"\tNon_Str Res:",len(muNup37_mol1[:]-muNup37_atomic1))

muNup37_mol2 = muNup37_mol1.create_copy( chain_id='a')
muNup37_atomic2 = muNup37_mol2.add_structure(muydimer, chain_id=muYDIMER['muNup37'][1],offset=0,ca_only = True)
muNup37_mol2.add_representation(muNup37_atomic2,resolutions=[beadsize_0,beadsize_3], color=rcolor)
muydimer_mols.append(muNup37_mol2)

print("Y_N37  : Str Res:",len(muNup37_atomic2),"\tNon_Str Res:",len(muNup37_mol2[:]-muNup37_atomic2))

rcolor = next(Rcolors);bcolor = rcolor #next(Bcolors)
muNup43_mol1 = st1.create_molecule('munup43', sequence=muYdimer_seqs['muNup43'], chain_id='A')
muNup43_atomic1 = muNup43_mol1.add_structure(muydimer, chain_id=muYDIMER['muNup43'][0],offset=0,ca_only = True)
muNup43_mol1.add_representation(muNup43_atomic1,resolutions=[beadsize_0,beadsize_3], color=rcolor)
muydimer_mols.append(muNup43_mol1)

print("Y_N43  : Str Res:",len(muNup43_atomic1),"\tNon_Str Res:",len(muNup43_mol1[:]-muNup43_atomic1))

muNup43_mol2 = muNup43_mol1.create_copy( chain_id='a')
muNup43_atomic2 = muNup43_mol2.add_structure(muydimer, chain_id=muYDIMER['muNup43'][1],offset=0,ca_only = True)
muNup43_mol2.add_representation(muNup43_atomic2,resolutions=[beadsize_0,beadsize_3], color=rcolor)
muydimer_mols.append(muNup43_mol2)

print("Y_N43  : Str Res:",len(muNup43_atomic2),"\tNon_Str Res:",len(muNup43_mol2[:]-muNup43_atomic2))

# --------------------------------------------------------------
# CREATING CLONES
# --------------------------------------------------------------

all_spoke_mol = []
all_spoke_mol += [mubskt_mols[0]]+[mubskt_mols[1]]
all_spoke_mol += [mubskt_mols[2]]+[mubskt_mols[3]]
all_spoke_mol += [mubskt_mols[4]]+[mubskt_mols[5]]

all_spoke_mol += muydimer_mols[:] 

num_bskt_mols = len(ytbskt_mols)+len(mubskt_mols)

NPC_MOLS = []

num_chains = len(chain_list['A'])

print("All one spoke:",all_spoke_mol)

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

		print("Mol:Chain:",nc,mol,mol_chain_id)

		#All clones of a particular protein should go into it
		newchain = chain_list[mol_chain_id][nc]
		clone = mol.create_clone(newchain)
		clone_mols[mol].append(clone)
		
for k,v in clone_mols.items():
	NPC_MOLS.append(v)


hier = s1.build()

#Write initial configuration
output = IMP.pmi.output.Output()
output.init_rmf(out_path+"/ini_all.rmf3", hierarchies=[hier])
output.write_rmf(out_path+"/ini_all.rmf3")
output.close_rmf(out_path+"/ini_all.rmf3")

mdl.update()  # propagates coordinates
#IMP.atom.show_with_representations(hier)

# --------------------------------------------------------------
# Setup degrees of freedom
# --------------------------------------------------------------

dof = IMP.pmi.dof.DegreesOfFreedom(mdl)

# --------------------------------------------------------------
#  RIGID BODIES from MLPs Coiled-Coil segments
# --------------------------------------------------------------

mlp1_added_rgbs=[];
mlp2_added_rgbs=[];

tpr1_added_rgbs=[];
tpr2_added_rgbs=[];

rigidbodyid = 0

def create_rigid_body_list(Prots_Rigid_Bodies, prot, startres, endres, rigidbodyid):
	"""Summary
	
	Args:
		Prots_Rigid_Bodies (TYPE): Input dictionary of proteins rigid bodies
		prot (TYPE): Protein of interest
		startres (TYPE): Start residue for rigid body
		endres (TYPE): Start residue for rigid body
		rigidbodyid (TYPE): Rigid body id to be set as value of the dictionary
	
	Returns:
		TYPE: Updated dictionary of proteins rigid bodies
	"""
	
	for res in list(range(startres,endres+1)):
		Prots_Rigid_Bodies[(prot,res)]=rigidbodyid

	return Prots_Rigid_Bodies


# --------------------------------------------------------------
#  RIGID BODIES from TPRs Coiled-Coil segments
# --------------------------------------------------------------

for nc, (rtmol1,rtmol2) in enumerate(zip(NPC_MOLS[0],NPC_MOLS[1])):
	
	for idx, ccseg in enumerate(tpr_cc_segments):
		print(idx, ccseg)
		
		mol1_rgb_sel_LR =IMP.atom.Selection(hier,
			molecule="tpr",
			resolution=IMP.atom.ALL_RESOLUTIONS,
			#resolution=beadsize_3,
			copy_index = 2*nc,
			residue_indexes=range(ccseg[0][0],ccseg[0][1]+1))

		#for tpr.0 and tpr.1 both are considered as tpr, as they joined to form a single rigid body
		rigidbodyid+=1 #Chnage the rigidbody id
		Prots_Rigid_Bodies = create_rigid_body_list(Prots_Rigid_Bodies, "tpr."+str(2*nc), ccseg[0][0], ccseg[0][1], rigidbodyid)
		
		mol2_rgb_sel_LR =IMP.atom.Selection(hier,
			molecule="tpr",
			resolution=IMP.atom.ALL_RESOLUTIONS,
			#resolution=beadsize_3,
			copy_index = 2*nc+1,
			residue_indexes=range(ccseg[1][0],ccseg[1][1]+1))

		Prots_Rigid_Bodies = create_rigid_body_list(Prots_Rigid_Bodies, "tpr."+str(2*nc+1), ccseg[1][0], ccseg[1][1], rigidbodyid)

		
		mol1_rgb_sel_LR_temp = []
		for rgbp in mol1_rgb_sel_LR.get_selected_particles():
			if rgbp in tpr1_added_rgbs:
				continue
			mol1_rgb_sel_LR_temp.append(rgbp)
			tpr1_added_rgbs.append(rgbp)

		mol2_rgb_sel_LR_temp = []
		for rgbp in mol2_rgb_sel_LR.get_selected_particles():
			if rgbp in tpr2_added_rgbs:
				continue
			mol2_rgb_sel_LR_temp.append(rgbp)
			tpr2_added_rgbs.append(rgbp)

			
		mol1_rgb_sel_densities =IMP.atom.Selection(hier,
			molecule="tpr",
			copy_index = 2*nc,
			#resolution=beadsize_0,
			residue_indexes=range(ccseg[0][0],ccseg[0][1]+1),
			representation_type=IMP.atom.DENSITIES)

		mol2_rgb_sel_densities =IMP.atom.Selection(hier,
			molecule="tpr",
			copy_index = 2*nc+1,
			#resolution=beadsize_0,
			residue_indexes=range(ccseg[1][0],ccseg[1][1]+1),
			representation_type=IMP.atom.DENSITIES)
		
		mol1_rgb_sel_LR_temp_dens = []
		for rgbp in mol1_rgb_sel_densities.get_selected_particles():
			if rgbp in tpr1_added_rgbs:
				continue
			mol1_rgb_sel_LR_temp_dens.append(rgbp)
			tpr1_added_rgbs.append(rgbp)

		mol2_rgb_sel_LR_temp_dens = []
		for rgbp in mol2_rgb_sel_densities.get_selected_particles():
			if rgbp in tpr2_added_rgbs:
				continue
			mol2_rgb_sel_LR_temp_dens.append(rgbp)
			tpr2_added_rgbs.append(rgbp)
		
		mol1_mol2_seg = mol1_rgb_sel_LR_temp + mol2_rgb_sel_LR_temp + mol1_rgb_sel_LR_temp_dens + mol2_rgb_sel_LR_temp_dens
		
		dof.create_rigid_body(rigid_parts=mol1_mol2_seg, max_trans=MC_TRANS)


# --------------------------------------------------------------
#  FLEXIBLE BEADS for Unstructured Regions for TPRs Proteins molecules
# --------------------------------------------------------------

for nc, (rtmol1,rtmol2) in enumerate(zip(NPC_MOLS[0],NPC_MOLS[1])):

	for idx, ccseg in enumerate(tpr_loop_segments):

		mol1_rgb_sel_LR =IMP.atom.Selection(hier,
			molecule="tpr",
			resolution=IMP.atom.ALL_RESOLUTIONS,
			#resolution=beadsize_4,
			copy_index = 2*nc,
			residue_indexes=range(ccseg[0][0],ccseg[0][1]+1)).get_selected_particles()

		mol2_rgb_sel_LR =IMP.atom.Selection(hier,
			molecule="tpr",
			resolution=IMP.atom.ALL_RESOLUTIONS,
			#resolution=beadsize_4,
			copy_index = 2*nc+1,
			residue_indexes=range(ccseg[1][0],ccseg[1][1]+1)).get_selected_particles()

		dof.create_flexible_beads(flex_parts=mol1_rgb_sel_LR,max_trans=MC_TRANS)

		dof.create_flexible_beads(flex_parts=mol2_rgb_sel_LR,max_trans=MC_TRANS)

# --------------------------------------------------------------
#  RIGID BODIES from Nup50
# --------------------------------------------------------------
for nc, (rtmol3,rtmol4) in enumerate(zip(NPC_MOLS[2],NPC_MOLS[3])):

	for nup50rgb in nup50_rigid_bodies:
	
		rtmol1_rgb_sel =IMP.atom.Selection(hier,
			molecule="nup50",
			resolution=IMP.atom.ALL_RESOLUTIONS,
			copy_index = 2*nc,
			residue_indexes=range(nup50rgb[0],nup50rgb[1]+1)).get_selected_particles()

		dof.create_rigid_body(rigid_parts=rtmol1_rgb_sel, max_trans=MC_TRANS)

		rigidbodyid+=1
		Prots_Rigid_Bodies = create_rigid_body_list(Prots_Rigid_Bodies, "nup50."+str(2*nc), nup50rgb[0],nup50rgb[1], rigidbodyid)

		rtmol2_rgb_sel =IMP.atom.Selection(hier,
			molecule="nup50",
			resolution=IMP.atom.ALL_RESOLUTIONS,
			copy_index = 2*nc + 1,
			residue_indexes=range(nup50rgb[0],nup50rgb[1]+1)).get_selected_particles()

		dof.create_rigid_body(rigid_parts=rtmol2_rgb_sel, max_trans=MC_TRANS)

		rigidbodyid+=1
		Prots_Rigid_Bodies = create_rigid_body_list(Prots_Rigid_Bodies, "nup50."+str(2*nc+1), nup50rgb[0],nup50rgb[1], rigidbodyid)

	dof.create_super_rigid_body(srb_parts=rtmol3.get_residues(),max_trans=MC_TRANS)
	dof.create_super_rigid_body(srb_parts=rtmol4.get_residues(),max_trans=MC_TRANS)

	# --------------------------------------------------------------
	#  FLEXIBLE BEADS from Nup50
	# --------------------------------------------------------------

	for nup50loop in nup50_nonrigid_residues:

		rtmol1_loop_sel =IMP.atom.Selection(hier,
			molecule="nup50",
			resolution=IMP.atom.ALL_RESOLUTIONS,
			copy_index = 2*nc,
			residue_indexes=range(nup50loop[0],nup50loop[1]+1)).get_selected_particles()

		dof.create_flexible_beads(flex_parts=rtmol1_loop_sel, max_trans=MC_TRANS)

		rtmol2_loop_sel =IMP.atom.Selection(hier,
			molecule="nup50",
			resolution=IMP.atom.ALL_RESOLUTIONS,
			copy_index = 2*nc+1,
			residue_indexes=range(nup50loop[0],nup50loop[1]+1)).get_selected_particles()

		dof.create_flexible_beads(flex_parts=rtmol2_loop_sel, max_trans=MC_TRANS)

# --------------------------------------------------------------
#  RIGID BODIES from Nup153
# --------------------------------------------------------------
for nc, (rtmol3,rtmol4) in enumerate(zip(NPC_MOLS[4],NPC_MOLS[5])):

	for nup153rgb in nup153_rigid_bodies:
		
		print(nc,rtmol3,rtmol4,nup153rgb)

		rtmol1_rgb_sel =IMP.atom.Selection(hier,
			molecule="nup153",
			resolution=IMP.atom.ALL_RESOLUTIONS,
			copy_index = 2*nc,
			residue_indexes=range(nup153rgb[0],nup153rgb[1]+1)).get_selected_particles()

		dof.create_rigid_body(rigid_parts=rtmol1_rgb_sel, max_trans=MC_TRANS)

		rigidbodyid+=1
		Prots_Rigid_Bodies = create_rigid_body_list(Prots_Rigid_Bodies, "nup153."+str(2*nc), nup153rgb[0],nup153rgb[1], rigidbodyid)

		rtmol2_rgb_sel =IMP.atom.Selection(hier,
			molecule="nup153",
			resolution=IMP.atom.ALL_RESOLUTIONS,
			copy_index = 2*nc+1,
			residue_indexes=range(nup153rgb[0],nup153rgb[1]+1)).get_selected_particles()

		dof.create_rigid_body(rigid_parts=rtmol2_rgb_sel, max_trans=MC_TRANS)

		rigidbodyid+=1
		Prots_Rigid_Bodies = create_rigid_body_list(Prots_Rigid_Bodies, "nup153."+str(2*nc+1), nup153rgb[0],nup153rgb[1],rigidbodyid)

	dof.create_super_rigid_body(srb_parts=rtmol3.get_residues(),max_trans=MC_TRANS)
	dof.create_super_rigid_body(srb_parts=rtmol4.get_residues(),max_trans=MC_TRANS)


	# --------------------------------------------------------------
	#  FLEXIBLE BEADS from Nup153
	# --------------------------------------------------------------

	for nup153loop in nup153_nonrigid_residues:

		rtmol1_loop_sel =IMP.atom.Selection(hier,
			molecule="nup153",
			resolution=IMP.atom.ALL_RESOLUTIONS,
			copy_index = 2*nc,
			residue_indexes=range(nup153loop[0],nup153loop[1]+1)).get_selected_particles()

		dof.create_flexible_beads(flex_parts=rtmol1_loop_sel, max_trans=MC_TRANS)

		rtmol2_loop_sel =IMP.atom.Selection(hier,
			molecule="nup153",
			resolution=IMP.atom.ALL_RESOLUTIONS,
			copy_index = 2*nc+1,
			residue_indexes=range(nup153loop[0],nup153loop[1]+1)).get_selected_particles()

		dof.create_flexible_beads(flex_parts=rtmol2_loop_sel, max_trans=MC_TRANS)


# write a single-frame RMF to view the model
out = IMP.pmi.output.Output()
out.init_rmf(out_path+"/initial_model.rmf3", hierarchies=[hier])
out.write_rmf(out_path+"/initial_model.rmf3")
out.close_rmf(out_path+"/initial_model.rmf3")

#################################################################

#All Spoke 1 mols (main spoke)
SPOKE_1_MOLS = []
for npcmols in NPC_MOLS:
	SPOKE_1_MOLS.append(npcmols[0])

SPOKE_1_MOLS_BSKT = []
for npcmols in NPC_MOLS[0:num_bskt_mols]:
	SPOKE_1_MOLS_BSKT.append(npcmols[0])

SPOKE_1_MOLS_BSKT_MU = []
for npcmols in [NPC_MOLS[0],NPC_MOLS[1],NPC_MOLS[2],NPC_MOLS[3],NPC_MOLS[4],NPC_MOLS[5]]: #tpr1, tpr2, nup50.0, nup50.1
	SPOKE_1_MOLS_BSKT_MU.append(npcmols[0])

SPOKE_1_MOLS_BSKT_MINUS_NUP153 = []
for npcmols in [NPC_MOLS[0],NPC_MOLS[1],NPC_MOLS[2],NPC_MOLS[3]]: #tpr1, tpr2, nup50.0, nup50.1
	SPOKE_1_MOLS_BSKT_MINUS_NUP153.append(npcmols[0])

SPOKE_1_MOLS_NUP153 = []
for npcmols in [NPC_MOLS[4],NPC_MOLS[5]]: #tpr1, tpr2, nup50.0, nup50.1
	SPOKE_1_MOLS_NUP153.append(npcmols[0])

#All spoke 1 mlp proteins (mlp1, mlp2)
SPOKE_1_MOLS_TPRS = []
for mlpmols in NPC_MOLS[0:2]:
	SPOKE_1_MOLS_TPRS.append(mlpmols[0])


SPOKE_1_MOLS_YDIM_MU = []
for npcmols in muydimer_mols:
	SPOKE_1_MOLS_YDIM_MU.append(npcmols)

# --------------------------------------------------------------
#  CONNECTIVITY RESTRAINTS FOR FIRST SPOKE ONLY EXCLUDING CLONES
# --------------------------------------------------------------

output_objects = []  # keep a list of functions that need to be reported

for mol in SPOKE_1_MOLS_BSKT_MINUS_NUP153:
	cr = IMP.pmi.restraints.stereochemistry.ConnectivityRestraint(mol)
	cr.add_to_model()
	output_objects.append(cr)

for mol in SPOKE_1_MOLS_NUP153:
	cr = IMP.pmi.restraints.stereochemistry.ConnectivityRestraint(mol)
	cr.add_to_model()
	output_objects.append(cr)



# --------------------------------------------------------------
#  MEMBRANE RESTRAINTS
# --------------------------------------------------------------

tor_th      = 45.0
tor_th_ALPS = 45.0
tor_R       = 465.0 + 150.0
tor_r       = 170.0 - tor_th/2.0

msl_sigma   = 1.0

if include_membrane_surface_binding:

	membrane_surface_sel = [
							(nup153_membrane_res[0][0],nup153_membrane_res[0][1],'nup153',0),
							(nup153_membrane_res[0][0],nup153_membrane_res[0][1],'nup153',1)]
	
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
		#msl_1.set_label('%s.%s'%(sel[2],sel[0]))
		output_objects.append(msl_1)
		print('Membrane binding restraint ready', msl_1.get_output())

# --------------------------------------------------------------
#  Z AXIAL RESTRAINTS
# --------------------------------------------------------------

if include_zaxial_position_restraint==True:

	nterm_idx = 0
	tpr1_nterm_res1 = 1
	tpr1_nterm_res2 = 150
	tpr2_nterm_res1 = 1
	tpr2_nterm_res2 = 150
	
	cterm_idx = -1
	tpr1_cterm_res = list(tpr_loop_segments)[cterm_idx][0][1]
	tpr2_cterm_res = list(tpr_loop_segments)[cterm_idx][1][1]

	tpr1_cterm_loopres_start = tpr_cc_segments[-4][0][1]+1
	tpr2_cterm_loopres_start = tpr_cc_segments[-4][1][1]+1

	tpr_cterm_extra_res = 50

	zapr_0 = IMP.pmi.restraints.npc.ZAxialPositionRestraint(
		hier,protein=(tpr_nbs_domain_loop[0],tpr_nbs_domain_loop[1],'tpr',0),
		lower_bound=-350,upper_bound=-300,
		weight=1.0)

	zapr_0.add_to_model()
	output_objects.append(zapr_0)

	zapr_0_1 = IMP.pmi.restraints.npc.ZAxialPositionRestraint(
		hier,protein=(tpr_nbs_domain_loop[0],tpr_nbs_domain_loop[1],'tpr',1),
		lower_bound=-350,upper_bound=-300,
		weight=1.0)

	zapr_0_1.add_to_model()
	output_objects.append(zapr_0_1)

	# TPR CTERM= AVERAGE=77nm +- 18 = ~700 to 950 A
	#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2174070/ 

	zapr_1 = IMP.pmi.restraints.npc.ZAxialPositionRestraint(
			hier,protein=(tpr1_cterm_loopres_start, tpr1_cterm_res,'tpr',0),
			lower_bound=-950,upper_bound=-700,
			weight=1.0)
	zapr_1.add_to_model()
	output_objects.append(zapr_1)

	zapr_2 = IMP.pmi.restraints.npc.ZAxialPositionRestraint(
			hier,protein=(tpr1_nterm_res1, tpr1_nterm_res2,'tpr',0),
			lower_bound=-950,upper_bound=-700,
			weight=1.0)
	zapr_2.add_to_model()
	output_objects.append(zapr_2)

	zapr_3 = IMP.pmi.restraints.npc.ZAxialPositionRestraint(
			hier,protein=(tpr2_cterm_loopres_start, tpr2_cterm_res,'tpr',1),
			lower_bound=-950,upper_bound=-700,
			weight=1.0)
	zapr_3.add_to_model()
	output_objects.append(zapr_3)

	#mlp mol 1 term='N' 10 residues
	zapr_4 = IMP.pmi.restraints.npc.ZAxialPositionRestraint(
			hier,protein=(tpr2_nterm_res1, tpr2_nterm_res2,'tpr',1),
			lower_bound=-950,upper_bound=-700,
			weight=1.0)
	zapr_4.add_to_model()
	output_objects.append(zapr_4)


# --------------------------------------------------------------
#  Occams Spring restaint
# --------------------------------------------------------------

if include_Occams:

	from StructuralEquivalenceRestraint import StructEquivalenceRestraint

	equiv_assignment_file = os.path.join(data_path,'equiv_assig_BSKT.dat')
	alignment_files_path = os.path.join(data_path,'alns')
	pdbs_files_path = os.path.join(data_path,'pdbs/')

	#Target distances file can be given as an input or generate based on the alignment
	target_complex_distances_file = os.path.join(pdbs_files_path,'target_complex_distances_file.dat')
	compute_target_distances_params = os.path.join(data_path,'target_distances_params.json')

	steqr1 = StructEquivalenceRestraint(hier, equiv_assignment_file, 
		alignment_files_path, pdbs_files_path,
		target_complex_distances_file=target_complex_distances_file, 
		compute_target_distances_params = compute_target_distances_params,
		rigid_bodies_list = Prots_Rigid_Bodies)


	output_objects = steqr1.set_restraint(output_objects)


# --------------------------------------------------------------
#  SAMPLING 1, REX1
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
	root_hier=hier ,
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


if include_EM_spec2:
	# Species 2
	tpr1_densities = IMP.atom.Selection(hier,molecule="tpr",copy_index = 0,
		representation_type=IMP.atom.DENSITIES).get_selected_particles()

	tpr2_densities = IMP.atom.Selection(hier,molecule="tpr",copy_index = 1,
		representation_type=IMP.atom.DENSITIES).get_selected_particles()

	densities = tpr1_densities + tpr2_densities
	
	print("Total Densities:",len(densities))

	gem = IMP.bayesianem.restraint.GaussianEMRestraintWrapper(densities, gmmfilename,
		scale_target_to_mass=True,slope=EM_SLOPE)
	gem.set_label("EM")
	gem.add_to_model()
	gem.set_weight(weight=EMR_WEIGHT)
	output_objects.append(gem)

	t0 = gem.evaluate()
	print('Eval. EM at t0: ', t0)
	#gmm_com = gem.get_center_of_mass()
	mdl.update()
	print("EM Restraint Eval:",gem.evaluate())

# --------------------------------------------------------------
#  EXCLUDED VOLUME RESTRAINTS
# --------------------------------------------------------------

evr1 = IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere(
	included_objects=SPOKE_1_MOLS_BSKT_MU,
	resolution=EV_RESOLUTION)
evr1.add_to_model()
output_objects.append(evr1)


evr2 = IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere(
	included_objects=SPOKE_1_MOLS_BSKT_MU,
	other_objects=SPOKE_1_MOLS_YDIM_MU,
	resolution=EV_RESOLUTION)
evr2.add_to_model()
output_objects.append(evr2)



# --------------------------------------------------------------
#  SAMPLING 2, REX2
# --------------------------------------------------------------

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



