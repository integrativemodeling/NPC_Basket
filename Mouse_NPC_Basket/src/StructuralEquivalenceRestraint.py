#!/usr/bin/env python
"""Summary
"""

#####################################################
# Last Update: March 26, 2024 by Neelesh Soni
# Andrej Sali group, University of California San Francisco (UCSF)
#####################################################


import IMP
import IMP.core
import IMP.pmi
import IMP.pmi.restraints

import os
import json
import numpy as np
from collections import OrderedDict
from pdbclass import PDB
import matplotlib.pyplot as plt



# -----------------------------------------
# Class for repulsive distance restraint on particles
# -----------------------------------------
class StructEquivalenceRestraint(IMP.pmi.restraints.RestraintBase):
	"""
	Constructor:
		Generate structural equivalence restraints between aligned residues.
	
	"""

	_include_in_rmf = True


	def __init__(self, root_hier, equiv_assignment_file, alignment_files_path, pdbs_files_path,
			target_complex_distances_file = None, 
			compute_target_distances_params = None,
			rigid_bodies_list = None,
			label = None, weight = 1.0):
		"""Summary
		
		Args:
			root_hier (TYPE): Root hierarchy
			equiv_assignment_file (TYPE): Equivalence assignments between alignments
			alignment_files_path (TYPE): alignment files path
			pdbs_files_path (TYPE): Input pdb files path
			target_complex_distances_file (None, optional): Input target complex distances
			compute_target_distances_params (None, optional): Input file for reading parameters
			rigid_bodies_list (None, optional): List of rigid bodies to include
			label (None, optional): Label for the restraint
			weight (float, optional): Weight of the restraint
		
		"""
		self.hier = root_hier

		model = self.hier.get_model()
		super(StructEquivalenceRestraint, self).__init__(
			model, name='StructEquivalenceRestraint', label=label, weight=weight)

		self.equiv_assignment_file = equiv_assignment_file
		self.alignment_files_path = alignment_files_path
		self.target_complex_distances_file = target_complex_distances_file
		self.compute_target_distances_params = compute_target_distances_params

		self.Structural_Equivalence=OrderedDict()
		self.Equivalence_Alignment = OrderedDict()
		self.Equivalence_Dist_Mat = OrderedDict()
		self.rigid_bodies_list = rigid_bodies_list

		self.read_equivalence_assignment()

		self.read_equiv_alignment_file(stepsize=10)
		
		if self.target_complex_distances_file!=None:
			self.Read_Distance_Matrix()
		else:

			with open(self.compute_target_distances_params,'r') as f:
				target_complex_params = json.load(f)

			for k,v in target_complex_params.items():
				if k == "protein_pdb_dict":
					target_complex_params[k] = v
				else:
					target_complex_params[k] = v.split(',')

			self.pdbs_files_path = pdbs_files_path
			self.protein_pdb_dict = target_complex_params["protein_pdb_dict"]
			self.prots_for_intradistance = target_complex_params["prots_for_intradistance"]
			self.skip_interprot_distances_for = target_complex_params["skip_interprot_distances_for"]
			
			for i, v in enumerate(self.skip_interprot_distances_for):
				self.skip_interprot_distances_for[i] = v.strip()
			
			self.DIST_CUTOFF1 = int(target_complex_params["DIST_CUTOFF1"][0])
			self.DIST_CUTOFF2 = int(target_complex_params["DIST_CUTOFF2"][0])

			self.DIST_CUTOFF1_PROT_LIST = target_complex_params["DIST_CUTOFF1_PROT_LIST"]
			self.DIST_CUTOFF2_PROT_LIST = target_complex_params["DIST_CUTOFF2_PROT_LIST"]

			for i, v in enumerate(self.DIST_CUTOFF1_PROT_LIST):
				self.DIST_CUTOFF1_PROT_LIST[i] = v.strip()
			for i, v in enumerate(self.DIST_CUTOFF2_PROT_LIST):
				self.DIST_CUTOFF2_PROT_LIST[i] = v.strip()

			self.tpr_seg1_seg2_midres = int(target_complex_params["tpr_seg1_seg2_midres"][0])

			for k, v in self.protein_pdb_dict.items():
				self.protein_pdb_dict[k] = [v.strip() for v in self.protein_pdb_dict[k].split(',')]

			self.Generate_Distance_Matrix()

			self.Read_Distance_Matrix()


		return


	def set_restraint(self, output_objects):
		"""Summary
		
		Args:
			output_objects (TYPE): Output_objects for the replica exchange class
		
		Returns:
			TYPE: Updated Output_objects
		"""
		equiv_restraints_added=[]

		distSD=15.0;
		beadsize_0=1

		for  protres_pair, dist in self.Equivalence_Dist_Mat.items():

			prot1, prot1res, prot2, prot2res = protres_pair
			
			prot1label = prot1.split('.')
			prot2label = prot2.split('.')

			try:
				prot_resi = IMP.pmi.restraints.basic.DistanceRestraint(self.hier,
							tuple_selection1=(prot1res, prot1res, prot1label[0], int(prot1label[1])),
							tuple_selection2=(prot2res, prot2res, prot2label[0], int(prot2label[1])),
							label = '-'.join(map(str,protres_pair)),
							resolution=beadsize_0,
							kappa=0.05,
							distancemin = dist-distSD ,
							distancemax = dist+distSD )
				
				prot_resi.add_to_model()
				output_objects.append(prot_resi)
				
				equiv_restraints_added.append((prot1, str(prot1res), prot2, str(prot2res), str(round(dist-distSD,2)), str(round(dist+distSD,2))))
			except:
				print("WARNING: protein pair not found!!",protres_pair)
				continue

		
		equiv_wlines=""
		for val in equiv_restraints_added:
			equiv_wlines+=' '.join(val)+'\n'
		with open('equiv_distances.dat','w') as outf:
			outf.writelines(equiv_wlines)

		print("Total number of Equivalence Restraints:",len(equiv_restraints_added))


		return output_objects

	def read_equivalence_assignment(self):
		"""Reads the structural equivalence file
		
		"""
		print('Reading structural equivalence file..')

		with open(self.equiv_assignment_file,'r') as inf:
			lines = inf.readlines()

			flag=0;key=None;
			for l in lines:
				
				if l[0]=='>':
					flag=1
					key=l[1:].strip()
					self.Structural_Equivalence[l[1:]]=[]

				if ((l[0]!='>') and (flag==1) ):
					
					self.Structural_Equivalence[key]=l.strip().split('/')

		assert len(self.Structural_Equivalence['spec0'])==len(self.Structural_Equivalence['spec1'])

		return


	def read_equiv_alignment_file(self, stepsize=1):
		"""Summary
		
		Args:
			stepsize (int, optional): Read restraint lines by skipping stepsize number of lines. Useful for reducing number of restraints
		
		"""
		for spec0_prot, spec1_prot in zip(self.Structural_Equivalence['spec0'],self.Structural_Equivalence['spec1']):

			spec0_prot_id, spec0_prot_copy = spec0_prot.split('.')
			spec1_prot_id, spec1_prot_copy = spec1_prot.split('.')
			
			spec0_spec1_fname = os.path.join(self.alignment_files_path,'mat_'+spec0_prot_id+'_'+spec1_prot_id+'.align')

			self.Equiv_aligned_pairs = OrderedDict()

			with open(spec0_spec1_fname,'r') as inf:
				lines = inf.readlines()
				print(spec0_spec1_fname,len(lines))
				for i,l in enumerate(lines):

					if i%stepsize!=0:
						continue

					toks = l.strip().split()
					self.Equiv_aligned_pairs[int(toks[2])]=[int(toks[5])]
			
			self.Equivalence_Alignment[spec0_prot,spec1_prot] = self.Equiv_aligned_pairs
		
		return


	def Read_Distance_Matrix(self):
		"""Summary
		
		"""
		print("Reading",self.target_complex_distances_file)

		with open(self.target_complex_distances_file,'r') as inf:
			lines = inf.readlines()

			for l in lines:
				
				toks = l.strip().split()

				prot1 = toks[0]
				prot1res = int(toks[1])
				prot2 = toks[2]
				prot2res = int(toks[3])
				dist = float(toks[4])

				self.Equivalence_Dist_Mat[tuple([prot1, prot1res, prot2, prot2res])] = dist

		return

	def check_protres_pair_rigidbody_identity(self, prot1_res1_pair, prot2_res2_pair):
		"""Check protein residue pairs are in the same rigid bodies
		
		Args:
			prot1_res1_pair (TYPE): Protein1 residue1 pair
			prot2_res2_pair (TYPE): Protein2 residue2 pair
		
		Returns:
			TYPE: True/False
		"""
		if (prot1_res1_pair) not in self.rigid_bodies_list.keys():
			return False

		if (prot2_res2_pair) not in self.rigid_bodies_list.keys():
			return False

		prot1_res1_pair_rgbid = self.rigid_bodies_list[(prot1_res1_pair)]
		prot2_res2_pair_rgbid = self.rigid_bodies_list[(prot2_res2_pair)]
		
		#return True if prot_res pair is in same rigid body
		if prot1_res1_pair_rgbid==prot2_res2_pair_rgbid:
			return True
		else:
			return False

		return


	def Generate_Distance_Matrix(self):
		"""Summary
		
		"""
		Equiv_Dist_Mat = OrderedDict()

		print("Species_Pair, Spec1_res, Spec2_res")
		for spec_pair, residue_pairs_dict in self.Equivalence_Alignment.items():
			
			prot_pdbfile = self.protein_pdb_dict[spec_pair[0]][0]
			prot_chainid = self.protein_pdb_dict[spec_pair[0]][1]

			#calling the PDB module to parse the "fname" and load it in memory
			mdl = PDB(os.path.join(self.pdbs_files_path+prot_pdbfile), residue_pairs_dict, [prot_chainid])
			
			#Error checking
			if mdl == None:
				print('ERROR: no such file:', prot_pdbfile)
				sys.exit(1)

			residue_pairs_dict = mdl.residue_pairs_dict

			for prot1_res, v in residue_pairs_dict.items():

				#skip residues that dont have structure. this is reflected in the v not having iter_id
				if len(v)!=2:
					continue

				iter_idx = v[1]
				prot2_res = v[0]
				prot1_coords = (mdl.x()[iter_idx],mdl.y()[iter_idx],mdl.z()[iter_idx])

				#Store coordinates of template proteins as coordinates of target protein. These coordinates will be used to get relative distances
				#Key is target protein and residue number
				Equiv_Dist_Mat[(spec_pair[1],prot2_res)]=prot1_coords


		Equiv_Dist_Matrix_file = []
		Equiv_Dist_Matrix_file2 = OrderedDict()


		Equiv_Dist_Matrix_file_proteins = []
		Equiv_Dist_Matrix_file_residues = []

		skipped_prots = []
		#Get distances:
		for i, ((prot1, prot1res), prot1_coords) in enumerate(Equiv_Dist_Mat.items()):

			Equiv_Dist_Matrix_file_proteins.append(prot1)
			Equiv_Dist_Matrix_file_residues.append(prot1res)

			Equiv_Dist_Mat_temp = []
			for j, ((prot2, prot2res), prot2_coords) in enumerate(Equiv_Dist_Mat.items()):

				if ((prot1 in self.skip_interprot_distances_for) and (prot2 in self.skip_interprot_distances_for)):
					Equiv_Dist_Mat_temp.append(0.0)
					continue	

				if (prot1==prot2) and (prot1 not in self.prots_for_intradistance):
					Equiv_Dist_Mat_temp.append(0.0)
					skipped_prots.append(prot1)
					continue

				#No effect of this "IF", when setting self.tpr_seg1_seg2_midres=-1
				if ((prot1.split('.')[0]=='tpr') and (prot2.split('.')[0]=='tpr')):

					#consider distances between same side of tpr segments. Consider only close pairs but within  1/3rds of TPRs
					if (prot1res < self.tpr_seg1_seg2_midres) and (prot2res > self.tpr_seg1_seg2_midres):
						continue
					elif (prot1res > self.tpr_seg1_seg2_midres) and (prot2res < self.tpr_seg1_seg2_midres):
						continue
					pass

				
				if self.check_protres_pair_rigidbody_identity((prot1, prot1res),(prot2, prot2res))==True:
					continue

				if i<j:
					
					dist = np.sqrt((prot1_coords[0]-prot2_coords[0])**2 + (prot1_coords[1]-prot2_coords[1])**2 + (prot1_coords[2]-prot2_coords[2])**2)

					#IF both proteins present in List1
					if ((prot1 in self.DIST_CUTOFF1_PROT_LIST) and (prot2 in self.DIST_CUTOFF1_PROT_LIST)):

						if dist < self.DIST_CUTOFF1:
							
							Equiv_Dist_Matrix_file2[((prot1, prot1res),(prot2, prot2res))] = [dist]

					#IF both proteins present in List2
					elif ((prot1 in self.DIST_CUTOFF2_PROT_LIST) and (prot2 in self.DIST_CUTOFF2_PROT_LIST)):

						if dist < self.DIST_CUTOFF2:
							
							Equiv_Dist_Matrix_file2[((prot1, prot1res),(prot2, prot2res))] = [dist]

					#IF proteins present in List1 and List2
					elif ((prot1 in self.DIST_CUTOFF1_PROT_LIST) and (prot2 in self.DIST_CUTOFF2_PROT_LIST)) or ((prot1 in self.DIST_CUTOFF2_PROT_LIST) and (prot2 in self.DIST_CUTOFF1_PROT_LIST)):

						if dist < self.DIST_CUTOFF2:
							
							Equiv_Dist_Matrix_file2[((prot1, prot1res),(prot2, prot2res))] = [dist]
		
		print("Intraproteindistances_skipped",set(skipped_prots))

		wlines = ""
		#Display top 10 distances
		for k in list(Equiv_Dist_Matrix_file2.keys())[:]:
			dist = Equiv_Dist_Matrix_file2[k][0]

			wlines+=str(k[0][0])+' '+str(k[0][1])+' '+str(k[1][0])+' '+str(k[1][1])+' '+str(round(dist,2))+'\n'

		self.target_complex_distances_file = self.pdbs_files_path+'target_complex_distances_file.dat'

		with open(self.target_complex_distances_file,'w') as outf:
			outf.writelines(wlines)

		return









