##standard imports
import numpy as np
import pickle
import scipy.stats
import os

from scipy.optimize import curve_fit
from lcfunctions import robustmean
import core

##installed things
from tqdm import tqdm
import bls
import mpyfit ##using mpyfit for the sliding window because its a C version and is incredibly fast compared to the python only version. Results are identical.
import batman ## Kreidbergs BATMAN package


'''
This file contains code for doing injection recovery modelling as in Rizzuto et al. 2017 and many of the ZEIT/THYME papers.
Separated from core because it requires other packages that a more casual user might not want to bother with installing.
'''

def construct_planet_signal(time,per,rp,b,t0,ecc,omega,mstar,rstar,oversample=20,exposure_time=30.0,ldpars=[0.4,0.3]):
    '''
    A Function that produces a model lightcurve from the BATMAN transit model package (Laura Kreidberg: https://www.cfa.harvard.edu/~lkreidberg/batman/).
    This is the basic model function that everything else will build from.
    INPUTS:
    ---------
        time (numpy array): the time points (in days) that you want model photometry for
        per: The orbital period in days for the model planet
        rp : planet radius in Earth radii
        b  : model impact parameter (0 - 1)
        t0 : time of mid transit, should make sense relative to your input time array.
        ecc: model eccentricity for the planet. Big numbers can cause numerical problems (best below 0.95).
        omega: model argument of periastron passage angle for the planet in degrees
        mstar: mass of the host star in Solar units
        rstar: radius of the host star in Solar units
    OPTIONALS:
    ----------
        oversample=20: the oversample rate in the model generation. See BATMAN package docs for details. 20 is usually ok.
        exposure_time=30.0: Exposure time for the observations you want to model, in minutes. This should match the data you want to compare the model to.
        ldpars=[0.4,0.3]: quadratic limb darkening parameters for the model star. Defaults are kind of of for K/M stars.
    OUTPUT:
    ---------
        batman_lc: A numpy array of the same size as the input time array containing the model lightcurve. By default it's baseline normalised to 1.
    '''
    
    ##exposure time in days
    time_exp = exposure_time/60.0/24.0
    
    ##convert planet radiuss in earth units to rp/rs
    rearth = 0.009154
    rprs = rp*rearth/rstar
    
    #convert to inclination and semimajor axis from period and impact_par using rstar,mstar
    semia = (mstar*(per/365.25)**2)**(1.0/3.0)*214.939469384/rstar
    inc   = np.arccos(b/semia)*180.0/np.pi ##in degrees as batman likes it
    #tdur  = per/np.pi*np.arcsin(np.sqrt(((1+rprs)**2 - b**2)/(1-np.cos(inc*np.pi/180)**2))/semia)*24.0 ##for testing
    
    ##set batman parameters
    params           = batman.TransitParams()
    params.t0        = t0
    params.per       = per
    params.rp        = rprs
    params.a         = semia
    params.inc       = inc
    params.ecc       = ecc
    params.w         = omega
    params.limb_dark = 'quadratic'
    params.u         = ldpars

    ##initialize batman
    batman_model = batman.TransitModel(params,time,supersample_factor=oversample,exp_time=time_exp)
    
    ##calculate the light curve
    batman_lc    = batman_model.light_curve(params)

    ##output the light curve
    return batman_lc
 
 
def inject_recover_tess(data,per,rp,impact,t0,ecc,omega,mstar,rstar,windowsize=1.0,demode=1,alias_num=2.0,min_period=1.00001,max_period=15.0,forcenull=False,exposuretime=None,ldpars=[0.4,0.3],oversample=20):
    '''
    Function that takes a single case, and does the injection/recovery test calling construct_planet_signal to build the injected model
    INPUTS:
    -------
        data (numpy recarray): in the format specified in interface.read_tess_data
        per: The orbital period in days for the model planet
        rp : planet radius in Earth radii
        impact: model impact parameter (0 - 1)
        t0 : time of mid transit, should make sense relative to your input time array.
        ecc: model eccentricity for the planet. Big numbers can cause numerical problems (best below 0.95).
        omega: model argument of periastron passage angle for the planet in degrees
        mstar: mass of the host star in Solar units
        rstar: radius of the host star in Solar units
    OPTIONALS:
    ----------
    windowsize=1.0: if demode=1 this is the nootch window size, if demode=2, this is the locor star rotation period.
    alias_num=2.0: as per locors alias_num,. the minimum period to alias the rotation period of the stars (windowsize) to.
    demode=1: 1=notch, 2=locor
    min_period=1.0: minimum bls search period in days
    max_period=15.0: maximum bls search period in days
    forcenull=False: If true, always will reject the notch model, for testing purposes.
    exposuretime=None: The exposure time in minutes of your input data. If None, will compute the most common time between data points and use that 
    ldpars=[0.4,0.3]: quadratic limb darkening parameters to use for star model
    oversample=20: oversample rate for batman, 20 is usually ok
    
    OUTPUTS:
    --------
    detected_it: 1 if recovered, 0 if not
    detp: recovered period if recovered
    dett0: recovered T0 if recovered
    detdp: recovered depth if recovered (not trustable, and meaningless if BIC search was run).
    '''
    
    ##if exposuretime is not provided
    if exposuretime is None: exposuretime = scipy.stats.mode(data.t-np.roll(data.t,1))[0][0]*24.0*60.0 ##just find the most common time between datapoints
    ##Build the synthetic transit signal with BATMAN
    injflux = construct_planet_signal(data.t,per,rp,impact,t0,ecc,omega,mstar,rstar,oversample=oversample,exposure_time=exposuretime,ldpars=ldpars)
    exp_depth = 1.0-np.min(injflux) ##The expected transit depth approximately at least
    ##copy so as not to overwrite original data
    udata=data.copy()
    ##now inject the planet signal into the rawdata aperture photometry, this happens to be easy
    udata.fcor = udata.fcor*injflux
    udata.fraw = udata.fraw*injflux
    
    ##now run the appropriate detrending algorithm
    if forcenull == False: 
        usedeltabic=-1.0
    else: usedeltabic = np.inf
    detected_it,detp,dett0,detdp = core.do_detrend(-1,-1,arclength=False, raw=False, wsize=windowsize,saveoutput=False,resolvabletrans=False,k2sff=False,indata=udata,period_matching = [per,exp_depth,t0],demode=demode,deltabic=usedeltabic,min_period=min_period,max_period = max_period,alias_num=alias_num,tess=True,show_progress=False)
    
    
    return detected_it,detp,dett0,detdp


def injrec_tesslist_mpi(tic,pointlist,rawdata,mstar,rstar,ids,thisrank=-1,machinename='laptop',jobname='',
                    wsize=1.0,demode=1,forcenull=False,alias_num=2.0,min_period=1.0001,max_period=12.0):
    '''
    THis function takes lists of parameters for trial planets and call inject_recover_tess multiple times, handling the output into a big array.
    See tess_injrec_mpi.py for usage example.
    Currently logging is commented out as its not general for everyone's system setup.
    
    '''
    ##open a log file to write to, not efficient but maybe worth it for bugshooting
    homedir = os.getenv("HOME")
    
    #if machinename == 'laptop': logfile = open('logfiles/logfile_main.txt','ab')
    #if machinename != 'laptop': logfile = open(homedir +'/project/K2pipe/logfiles/logfile_'+jobname+'_' + str(thisrank)+'.txt','ab')
    #logfile.write(str(thisrank) + ' ' + str(epic) + ' \n')
    #logfile.flush()
    for i in range(pointlist.shape[0]):
    #for i in range(1): ##for testing

        #logfile.write(str(i)+'_'+str(thisrank) +' ' +str(pointlist[i,0]) + ' ' +str(pointlist[i,1])+ ' ' +str(pointlist[i,2])+ ' ' +str(pointlist[i,3]) + ' ' +str(pointlist[i,4]) + ' ' +str(pointlist[i,5]) + ' ' +str(pointlist[i,6]) + ' \n')
        #logfile.flush()
        
        ##run the inject_test on this particular planet, wrap in error catcher so that exceptions can be diagnostic
        ##The code with exception catch
        thisdetect,detp,dett0,detdp = inject_recover_tess(rawdata,pointlist[i,0],pointlist[i,1],pointlist[i,2],pointlist[i,3]*pointlist[i,0]+rawdata.t[0],pointlist[i,4], pointlist[i,5],mstar,rstar,windowsize=wsize,demode=demode,forcenull=forcenull,alias_num=alias_num,min_period=min_period,max_period=max_period)
        # except Warning: ##return the information needed to figure out what went wrong, and the error flag
#             thestring = str(pointlist[i,0])+' ,'+str(pointlist[i,1])+' ,'+str(pointlist[i,2])+' ,'+str(pointlist[i,3])+' ,'+str(pointlist[i,4])+' ,'+str(pointlist[i,5])+' ,'+str(pointlist[i,6])
#             thestring = 'rank:'+ids+' PPARS:'+thestring
#             return thestring, -1
#         ##make it known we reached the last one
        #logfile.write('ddd')
        #logfile.flush()
            
       ##Everything is fine, output the right way
        wedet=0
        if thisdetect == True:  wedet = 1
        pointlist[i,6] = wedet
        pointlist[i,7] = detp
        pointlist[i,8] = dett0
        pointlist[i,9] = detdp
    
    ##logfile.close
    return pointlist, 1
    
def smooth(x,window_len,win='flat',simsize=True):

##pad the signal:
    s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]

##make up the window
    if not win in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']: 
        print('I dont known that type of window bro')
        print('Use: flat, hanning, hamming, bartlett, blackman')
        pdb.set_trace()
    
    if win == 'flat':
        w=np.ones(window_len,'d') ##makes a moving average over this window length
    else:
        w=eval('np.'+win+'(window_len)')



    y=np.convolve(w/w.sum(),s,mode='valid')
    
    if simsize == True:
        y = y[int((window_len-1)/2):int(len(y)-(window_len-1)/2)]

    return y    
    
def plot_ijresults(files,targetname = '',outdir = './',detected_planets = [],extratag = '',numbins=15, oldfile=False)    :

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    import pickle
    import os, glob
    from scipy.interpolate import NearestNDInterpolator as nninterp
    from matplotlib import ticker


    mpl.rcParams['lines.linewidth']   = 1.5
    mpl.rcParams['axes.linewidth']    = 2
    mpl.rcParams['xtick.major.width'] =2
    mpl.rcParams['ytick.major.width'] =2
    mpl.rcParams['ytick.labelsize'] = 15
    mpl.rcParams['xtick.labelsize'] = 15
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['legend.numpoints'] = 1
    mpl.rcParams['axes.labelweight']='semibold'
    mpl.rcParams['axes.titlesize']=16
    mpl.rcParams['axes.titleweight']='semibold'
    mpl.rcParams['font.weight'] = 'bold'
    rearth =  0.009154

    start = 0


    for i in range(len(files)):
        ijtup = ()
        #pdb.set_trace()
        
        if oldfile == False:ijlist,thisepic,cnum = pickle.load(open(files[i],'rb'))
        if oldfile == True: ijlist,thisepic,cnum = pickle.load(open(files[i],'rb'),encoding='latin1')
       

        namestr = files[i].split('/')[1].split('.pkl')[0]  + '_'+extratag  
        if targetname != '': namestr=targetname
        if extratag != '' : namestr  += '_'+extratag



        fig1,ax1 = plt.subplots(1)
        for j in range(len(ijlist)):
            fmt = '.r'
            if ijlist[j,6] == 1: fmt = '.b'
            ax1.plot(ijlist[j,0],ijlist[j,1],fmt,alpha=0.8)
        ax1.set_xlabel('P (days)')
        ax1.set_ylabel('R '+r'(R$_\mathbf{\mathrm{E}}$)')

        plims = [np.floor(np.min(ijlist[:,0])),np.ceil(np.max(ijlist[:,0]))]
        rlims = [np.floor(np.min(ijlist[:,1]*10))/10.0,np.ceil(np.max(ijlist[:,1]))]

        rlims = [np.floor(np.min(ijlist[:,1]*10))/10.0,10.0]

        ax1.set_xlim(plims)
        ax1.set_ylim(rlims)
        fig1.suptitle('EPIC '+str(thisepic),fontsize=18,fontweight='semibold')
        fig1.savefig(outdir + str(thisepic)+'_'+str(i)+'_'+namestr+'_pointwise.pdf')
        plt.close(fig1)

        found = np.where(ijlist[:,6]  > 0.9)[0]
        missed = np.where(ijlist[:,6] < 0.5)[0]
        nbin = numbins
        thisto,txedge,tyedge = np.histogram2d(ijlist[:,0],ijlist[:,1],bins=nbin,range=[plims,rlims])
        fhisto,fxedge,fyedge =  np.histogram2d(ijlist[found,0],ijlist[found,1],bins=nbin,range=[plims,rlims])
        rfrac = fhisto/thisto
        binx = fxedge[0:len(fxedge)-1] + fxedge[1]/2. - fxedge[0]/2.
        biny = fyedge[0:len(fyedge)-1] + fyedge[1]/2. - fyedge[0]/2.
        xp,yp = np.meshgrid(binx,biny) 
        xx = xp.flatten()
        yy = yp.flatten()
        rr = rfrac.flatten()

        qwe = np.where(np.isnan(rr)==False)[0]
        xx = xx[qwe]
        yy = yy[qwe]
        rr = rr[qwe]
        pointlist = np.zeros((len(xx),2),float)
        pointlist[:,0] = xx
        pointlist[:,1] = yy
    
        myinterp = nninterp(pointlist,rr)
        xi = np.linspace(plims[0],plims[1],100)
        yi = np.linspace(rlims[0],rlims[1],100)
        xgrid,ygrid = np.meshgrid(xi,yi)
        zz = ygrid*0.0
        for ii in range(100):
            for jj in range(100):
                #pdb.set_trace()
                zz[ii,jj] = myinterp(xgrid[ii,jj],ygrid[ii,jj])

        fig,ax = plt.subplots()
        plt.imshow(zz.T,origin='lower',aspect='auto',interpolation='none',extent=[plims[0],plims[1],rlims[0],rlims[1]],cmap='cubehelix')
        cb= plt.colorbar()# (generate plot here)
    
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()
        cb.ax.tick_params(axis='y', direction='out')

        plt.xlabel('P (days)')
        plt.ylabel('R '+r'(R$_\mathbf{\mathrm{E}}$)')
        ax.plot(10.18,3.3,'*',markerfacecolor='lime',markeredgecolor='k',markeredgewidth=2,markersize=25)
        #ax.plot(20.54,2.62,'*',markerfacecolor='lime',markeredgecolor='k',markersize=25,markeredgewidth=2)
        #ax.plot(6.96,9.9,'*w',markeredgecolor='k',markersize=15)
        labelname = 'TIC' + str(thisepic)
        if targetname != '' : labelname = targetname
        ax.text(1.6,9.2,labelname,backgroundcolor='w')
        ax.tick_params(direction='out')

        plt.tight_layout()
        plt.gcf().subplots_adjust(left=0.15)
        plt.savefig(outdir + str(thisepic)+'_'+str(i)+'_'+namestr+'_interpolation.pdf')
    
        ##now make one that has both things on the plot
        for j in range(len(ijlist)):
            fmt = '.r'
            if ijlist[j,6] == 1: fmt = '.b'
            if j % 5 == 0: ax.plot(ijlist[j,0],ijlist[j,1],fmt,alpha=0.8,)
        ax.set_xlim(plims)
        ax.set_ylim(rlims)
        for pl in detected_planets:
            ax.plot(pl[0],pl[1],'*',markerfacecolor='lime',markeredgecolor='k',markeredgewidth=2,markersize=25)
        
        labelname = 'TIC' + str(thisepic)
        if targetname != '' : labelname = targetname
        ax.text(1.6,9.2,labelname,backgroundcolor='w')
        ax.tick_params(direction='out')

        plt.tight_layout()
        plt.gcf().subplots_adjust(left=0.15)

        plt.savefig(outdir + str(thisepic)+'_'+str(i)+'_'+namestr+'_combo.pdf')


        plt.close("all")

        ##now figure out a confidence interval line at some level: e.g. 90%?
        levels = [0.5,0.8,0.9]

        plist = xgrid[0]*1.0
        rgrid = ygrid[:,0]*1.0
        lims= np.zeros((len(plist),len(levels)),dtype=float)
        for i in range(len(plist)):
            theselims = np.interp(levels,zz[i],rgrid)
            lims[i] = theselims*1.0
        
       
    
    
    
        #pdb.set_trace()
        figl,axl = plt.subplots()
        axl.plot(plist,smooth(lims[:,0],9),'k',label='50%') 
        axl.plot(plist,smooth(lims[:,1],9),'b',label='80%')
        axl.plot(plist,smooth(lims[:,2],9),'r',label='90%')
        axl.legend()
        axl.set_ylabel(r'Planet Radius (R$_\oplus$)')
        axl.set_xlabel('Planet Period (days)')
        plt.tight_layout()
        figl.savefig(outdir + str(thisepic)+'_'+str(i)+'_'+namestr+'_simplelims.pdf')
        
        l10   = np.where(plist <= 10)[0]
        l1020 = np.where((plist > 10) & (plist < 20))[0]
        l2030 = np.where((plist >= 20) & (plist < 25))[0]
    
        lim10 = np.mean(lims[l10],axis=0)
        lim20 = np.mean(lims[l1020],axis=0)
        lim30 = np.mean(lims[l2030],axis=0)
    
        plt.close('all')


    

        
    
    

    print('DONE')


    
    
    
    