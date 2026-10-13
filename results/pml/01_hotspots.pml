# Hotspot selection checkpoint
load /home/payman/projects/learning/compbio-bioinfx-master/drug-discovery/inputs/pdbs/insulin_target.pdb, target
hide everything
show surface, target
show cartoon, target
color grey80, target
set transparency, 0.65, target
select hotspots, target and chain A and resi 59+83+91
show sticks, hotspots
color red, hotspots
set label_size, 16
set label_color, white
label hotspots and name CA, "%s%s" % (resi, resn)
orient hotspots
zoom hotspots, 8
set ray_shadows, 0
