import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl 
import pdb
mpl.rcParams['lines.linewidth']   = 2
mpl.rcParams['axes.linewidth']    = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.labelsize']   = 10
mpl.rcParams['xtick.labelsize']   = 10
mpl.rcParams['axes.labelsize']    = 10
mpl.rcParams['legend.numpoints']  = 1
mpl.rcParams['axes.labelweight']  = 'semibold'
mpl.rcParams['axes.titlesize']    = 10
mpl.rcParams['axes.titleweight']  = 'semibold'
mpl.rcParams['font.weight']       = 'semibold'

import interface as thymeNL
import lcfunctions as lcfunc
import injection_recovery as IJ


target_name = 166527623
datafile = 'tessdata/tess2019112060037-s0011-0000000166527623-0143-s_lc.fits'

hip67522 = thymeNL.target(target_name)
hip67522.load_data_tess(datafile)

per   = 5
rp    = 10.0
impact= 0.0
ecc   = 0.0
t0    = hip67522.data.t[0]+1.0
omega = 90.0
mstar = 1.22
rstar = 1.38

dummy = IJ.inject_recover_tess(hip67522.data,per,rp,impact,t0,ecc,omega,mstar,rstar,windowsize=0.5,demode=1,alias_num=2.0,min_period=1.00001,max_period=15.0,forcenull=False,exposuretime=None,ldpars=[0.4,0.3],oversample=20)





pdb.set_trace()
print('Done')