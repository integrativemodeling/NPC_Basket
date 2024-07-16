
# This file provides information about how to generate coconut alignments

# git clone git@github.com:neeleshsoni21/COCONUT.git
# cd COCONUT
#
# mkdir coconut/example/pipeline1/tpr
# Download fasta file from uniprot (https://www.uniprot.org/uniprotkb/F6ZDS4/entry) and add to the tpr folder
# Download pcoil probability file from https://toolkit.tuebingen.mpg.de/tools/pcoils and add to the tpr folder


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
# For TPR-TPR alignment.
#---------------------------------------------------#
data_dir = 'tpr'
protein1, protein1_pcoils = 'TPR','pcoils_tpr.out'
protein2, protein2_pcoils = 'TPR','pcoils_tpr.out'
example1(data_dir, protein1, protein1_pcoils, protein2, protein2_pcoils)


