# Monomer self-consistency checkpoint (MANIFEST Section 4)
load /home/payman/projects/learning/compbio-bioinfx-master/drug-discovery/results/01_backbones/design_0.pdb, design
load /home/payman/projects/learning/compbio-bioinfx-master/drug-discovery/results/01_backbones/design_0.pdb, predicted
hide everything
show cartoon, design or predicted
color cyan, design
color magenta, predicted
# cycles=0 disables outlier rejection, for comparability with scRMSD
align predicted, design and chain B, cycles=0
orient design
set ray_shadows, 0
print "PyMOL align RMSD above is an independent check on metrics.scrmsd"
