import interface as thymeLN
import core
import matplotlib.pyplot as plt


target_name = 166527623
dfile = 'tessdata/tess2019112060037-s0011-0000000166527623-0143-s_lc.fits'

targ = thymeLN.target(target_name)
targ.load_data_tess(dfile)


##notch
targ.run_notch()


##locor
#targ.prot = 1.4
#targ.run_locor()


import pdb
pdb.set_trace()
