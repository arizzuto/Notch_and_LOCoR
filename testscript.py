import interface as thymeLN
import core,lcfunctions
import matplotlib.pyplot as plt
import pickle

target_name = 166527623
dfile = 'tessdata/tess2019112060037-s0011-0000000166527623-0143-s_lc.fits'

targ = thymeLN.target(target_name)
targ.load_data_tess(dfile)


##notch
targ.run_notch()

fff = open('hip67522_test.pkl','wb')
pickle.dump(targ,fff)
fff.close()

targ = pickle.load(open('hip67522_test.pkl','rb'))
##locor
#targ.prot = 1.4
#targ.run_locor()


#best_px,dpx,t0x,detsigx,firstpower,pgrid,firstphase,dcycx  = lcfunctions.run_bls(targ.data,targ.notch.detrend,targ.notch.badflag,rmsclip=1.5,snrcut=7.0,searchmax=15.0,searchmin=1.00001,binn=300,mindcyc=0.005,maxdcyc=0.3,freqmode='standard')
#
best_p,dp,t0,detsig,firstpower2,pgrid2,firstphase2,dcyc  = lcfunctions.run_bls_bic(targ.data,targ.notch.bicstat,targ.notch.badflag,rmsclip=1.5,snrcut=7.0,searchmax=15.0)




import pdb
pdb.set_trace()


plt.plot(targ.data.t,targ.data.fcor,'.')
