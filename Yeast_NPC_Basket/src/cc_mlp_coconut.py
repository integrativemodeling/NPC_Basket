
# This file provides information about how to generate coconut alignments

# git clone git@github.com:neeleshsoni21/COCONUT.git
# cd COCONUT
#
# mkdir coconut/example/pipeline1/mlp
# Download mlp1 fasta file from uniprot (https://www.uniprot.org/uniprotkb/Q02455/entry) and add to the mlp folder
# Download mlp2 fasta file from uniprot (https://www.uniprot.org/uniprotkb/P40457/entry) and add to the mlp folder
# Download pcoil probability file from https://toolkit.tuebingen.mpg.de/tools/pcoils and add to the mlp folder


# Re-train coconut models if necessary or pickle version error. 
# Input data is provided in the distribution. Training takes about few minutes
# Train using the following code
#---------------------------------------------------#
# Re-training coconut models
#---------------------------------------------------#
# import coconut as cc
# cc_t_obj = cc.COCONUT()
# cc_t_obj.train_coconut_models()


#---------------------------------------------------#
# For generating COCONUT alignments
#---------------------------------------------------#
import coconut as cc

from pipeline1.load_pipeline import example1

#---------------------------------------------------#
# For MLP1-MLP1 alignment.
#---------------------------------------------------#
data_dir = 'mlp'
protein1, protein1_pcoils = 'MLP1','pcoils_mlp1.out'
protein2, protein2_pcoils = 'MLP1','pcoils_mlp1.out'
example1(data_dir, protein1, protein1_pcoils, protein2, protein2_pcoils)


#---------------------------------------------------#
# For MLP1-MLP2 alignment.
#---------------------------------------------------#
data_dir = 'mlp'
protein1, protein1_pcoils = 'MLP1','pcoils_mlp1.out'
protein2, protein2_pcoils = 'MLP2','pcoils_mlp2.out'
example1(data_dir, protein1, protein1_pcoils, protein2, protein2_pcoils)

