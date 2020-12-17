import interface as thymeLN
import core,lcfunctions,vetting
import matplotlib.pyplot as plt
import numpy as np

import pdb
import pickle

# target_name = 166527623
# dfile = 'tessdata/tess2019112060037-s0011-0000000166527623-0143-s_lc.fits'

# target_name = 159873822
# 
# dfile1 = 'tessdata/TOI2048/tess2019253231442-s0016-0000000159873822-0152-s_lc.fits'
# sect1 = '16'
# 
# dfile2 = 'tessdata/TOI2048/tess2020078014623-s0023-0000000159873822-0177-s_lc.fits'
# sect2 = '23'
# 
# dfile3 = 'tessdata/TOI2048/tess2020106103520-s0024-0000000159873822-0180-s_lc.fits'
# sect3 = '24'
# 
# 
# 
# targ1 = thymeLN.target(target_name)
# targ1.load_data_tess(dfile1,sector_label=sect1)
# 
# 
# targ2 = thymeLN.target(target_name)
# targ2.load_data_tess(dfile2,sector_label=sect2)
# 
# targ3 = thymeLN.target(target_name)
# targ3.load_data_tess(dfile3,sector_label=sect3)
# 
# ##notch
# # targ1.run_notch()
# # targ2.run_notch()
# # targ3.run_notch()
# # 
# # 
# # fff = open('toi2048_test_16-23-24.pkl','wb')
# # pickle.dump((targ1,targ2,targ3),fff)
# # fff.close()
# 
# targ1,targ2,targ3 = pickle.load(open('toi2048_test_16-23-24.pkl','rb'))
# 
# 
# 
# 
# ctarg0 = targ1.combine_lc(targ2)
# ctarg  = ctarg0.combine_lc(targ3)
# 
# pdb.set_trace()
# ppp = core.calc_phase(ctarg.data.t,13.79,1739.1133+13.79/2)
# intr = np.where((ppp > 0.49) & (ppp<0.51))[0]
##locor
#targ.prot = 1.4
#targ.run_locor()

#ctarg = pickle.load(open('hip67522_test.pkl','rb'))  

target_name='dstuc'
dfile = 'tessdata/tess2018206045859-s0001-0000000410214986-0120-s_lc.fits'
sect = '1'
# 
# ctarg = thymeLN.target(target_name)
# ctarg.load_data_tess(dfile,sector_label=sect)
# 
# ctarg.run_notch()

# fff=open('dstuc_test.pkl','wb')
# pickle.dump(ctarg,fff)
# fff.close()

ctarg =pickle.load(open('dstuc_test.pkl','rb'))


#best_px,dpx,t0x,detsigx,firstpower,pgrid,firstphase,dcycx  = lcfunctions.run_bls(targ1.data,targ1.notch.detrend,targ1.notch.badflag,rmsclip=1.5,snrcut=7.0,searchmax=15.0,searchmin=1.00001,binn=300,mindcyc=0.005,maxdcyc=0.3,freqmode='standard')
#

best_p,dp,t0,detsig,firstpower2,pgrid2,firstphase2,dcyc  = lcfunctions.run_bls_bic(ctarg.data,ctarg.notch.bicstat,ctarg.notch.badflag,rmsclip=1.5,snrcut=7.0,searchmax=15.0,searchmin=1.00001)

best_px,dpx,t0x,detsigx,firstpower,pgrid,firstphase,dcycx  = lcfunctions.run_bls(ctarg.data,ctarg.notch.detrend,ctarg.notch.badflag,rmsclip=1.5,snrcut=7.0,searchmax=15.0,searchmin=1.00001,binn=300,mindcyc=0.005,maxdcyc=0.3,freqmode='standard')

prot=1.41
##pdb.set_trace()
eofig,eoax = vetting.eo_plot(ctarg.data,ctarg.notch.detrend,ctarg.notch.badflag,best_p[0],t0[0],dcyc[0],id=ctarg.id,outdir='')


import pdb
pdb.set_trace()


plt.plot(targ.data.t,targ.data.fcor,'.')
