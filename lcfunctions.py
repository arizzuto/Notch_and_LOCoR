import numpy as np
import core
from lightkurve import search_lightcurvefile #to be deprecated soon, should use search_lightcurvefile. will need to update time array
import os
import interface as thymeNL #is this circular and bad?
import pickle as pkl

def LCconvertCTL(lcdata,removebad=True):
    ##convert to notch pipeline format inputs:
    ### This takes a tess 2min cadence lightcurve (from the bulk DL page for example)
    ### And formats it for notch
    
    dl = len(lcdata)
    outdata         = np.recarray((dl,),dtype=[('t',float),('fraw',float),('fcor',float),('s',float),('qual',int),('divisions',float)])
    outdata.t       = lcdata['TIME']
    outdata.fraw    = lcdata['SAP_FLUX']
    outdata.fcor    = lcdata['PDCSAP_FLUX']
    outdata.qual    = lcdata['QUALITY']
    okok = np.where(np.isnan(outdata.fcor)==False)[0]
    outdata.fraw /= np.nanmedian(outdata.fraw)
    outdata.fcor /= np.nanmedian(outdata.fcor)
    outdata   = outdata[okok]
    keep      = np.where(outdata.qual == 0)[0]
    outdata.s = 0.0
    outdata   = outdata[keep]
    return outdata    
    

def LCcombine(lclist):
    ##combine multiple stacks of lightcurves in format created by LCconvertCTL above, and interface.py
    ##lclist is just a tuple of such objects
    nsect = len(lclist)
    from numpy.lib.recfunctions import stack_arrays
    outdata = stack_arrays(lclist,asrecarray= True,usemask = False)
    return outdata
    
def robustmean(y,cut):     
    '''
    Compute a robust mean of a variable, used in other functions
    '''
    absdev    = np.absolute(y-np.median(y))
    medabsdev = np.median(absdev)/0.6745
    if medabsdev < 1.0e-24: medabsdev = np.mean(absdev)/0.8
    gind = np.where(absdev <= cut*medabsdev)[0]
    sigma    = np.std(y[gind])
    sc       = np.max([cut,1.0])
    
    if sc <= 4.5: sigma = sigma/(-0.15405+0.90723*sc-0.23584*sc**2+0.020142*sc**3) 
    goodind  = np.where(absdev <= cut*sigma)[0]
    #goodpts  = y[goodind]
    sigma    = np.std(y[goodind])
    
    sc = np.max([cut,1.0])
    if sc <= 4.5: sigma=sigma/(-0.15405+0.90723*sc-0.23584*sc**2+0.020142*sc**3)      
    return np.mean(y[goodind]),sigma/np.sqrt(len(y)-1.0),len(y) - len(goodind),goodind
    
    
    
def run_notch(data,window=0.5,mindbic=-1.0,useraw=False):
    '''
        a wrapper that runs the Notch filter detrending via core.do_detrend()
    '''
    
    print('Running notch filter pipeline with windowsize of ' + str(window) + ' and minimum DeltaBIC of ' +str(mindbic))    
    fittimes,depth,detrend,polyshape,badflag = core.do_detrend(1,1001,arclength=False,raw=useraw,wsize=window,indata=data,saveoutput=False,outdir='',resolvabletrans=False,demode=1,cleanup=True,deltabic=mindbic)
    
    ##Store everything in a common format recarray
    dl=len(detrend)
    notch         = np.recarray((dl,),dtype=[('t',float),('detrend',float),('polyshape',float),('notch_depth',float),('deltabic',float),('bicstat',float),('badflag',int)])
    notch.t = data.t
    notch.notch_depth = depth[0].copy()
    notch.deltabic    = depth[1].copy() 
    notch.detrend     = detrend.copy()
    notch.badflag     = badflag.copy()
    notch.polyshape   = polyshape.copy()
    
    bicstat = notch.deltabic-np.median(notch.deltabic)
    notch.bicstat = 1- bicstat/np.max(bicstat)

    return notch
    
def run_locor(data,prot,alias_num=0.01,useraw=False):
    '''
        a wrapper that runs the LOCoR detrending via core.do_detrend()
    '''
    print('Running LOCoR with rotation period of ' + str(prot) + ' and alias_num of ' + str(alias_num))
    fittimes,depth,detrend,polyshape,badflag = core.do_detrend(1,1001,arclength=False,raw=useraw,wsize=prot,indata=data,saveoutput=False,outdir='',resolvabletrans=False,demode=2,alias_num=alias_num,cleanup=True)
    
    ##store everything in a common format recarray
    dl=len(detrend)
    locor         = np.recarray((dl,),dtype=[('t',float),('detrend',float),('badflag',int)])
    locor.detrend = detrend.copy()
    locor.badflag = badflag.copy()
    locor.t = data.t
    
    return locor
    
def run_bls(data,targetlc,badflag,rmsclip=1.5,snrcut=7.0,searchmax=20.0,searchmin=1.00001,binn=300,mindcyc=0.005,maxdcyc=0.3,freqmode='standard'):
    '''
    A wrapper that runs the iterative BLS search via core.bls_transit_search
    parameters are set for generally use
    '''
    best_px,dpx,t0x,detsigx,firstpower,pgrid,firstphase,dcycx =  core.bls_transit_search(data,targetlc,badflag,rmsclip=rmsclip,snrcut=snrcut,searchmax=searchmax,searchmin=searchmin,binn=binn,period_matching=[-1,-1,-1],mindcyc=mindcyc,maxdcyc=maxdcyc,freqmode=freqmode,datamode='standard')
    return best_px,dpx,t0x+best_px/2.0,detsigx,firstpower,pgrid,firstphase,dcycx  ##the change to t0 here makes transits at phase 0
    
def run_bls_bic(data,targetbic,badflag,rmsclip=1.5,snrcut=7.0,searchmax=20.0,searchmin=1.00001,binn=300,mindcyc=0.003,maxdcyc=0.01,freqmode='standard'):
    '''
    A wrapper that runs the iterative BLS search on the BIC statistic via core.bls_transit_search
    parameters set for BIC specifically
    '''
    
    best_px,dpx,t0x,detsigx,firstpower,pgrid,firstphase,dcycx =  core.bls_transit_search(data,targetbic,badflag,rmsclip=rmsclip,snrcut=snrcut,searchmax=searchmax,searchmin=searchmin,binn=binn,period_matching=[-1,-1,-1],mindcyc=mindcyc,maxdcyc=maxdcyc,freqmode=freqmode,datamode='bic')
    return best_px,dpx,t0x+best_px/2.0,detsigx,firstpower,pgrid,firstphase,dcycx 


def showmenotch(ontr=True):
    import pdb
    '''
    Function that draws plots to explain what Notch does.
    '''
    
    ##generate 1 day of fake lightcurve.
    t = np.arange(0,1,2.0/60.0/24.0)
    ##the lightcurve, lets make it a parabola
    lc = -(t*0.9-0.5)*(t*0.9-0.5)/100
    lc = lc/np.median(lc)
    lc = lc/np.max(lc)/10
    lc = lc-np.median(lc)
    polymod = lc*1.0 + 1.0
    lc = lc + np.random.normal(1.0,0.003,len(lc))
    ##add some other wiggles
    wiggles  = np.sin(t*10*2*np.pi)*0.001
    wiggles += np.sin(t*5*2*np.pi)*0.001
    wiggles += np.sin(t*2*2*np.pi)*0.002
    lc = lc + wiggles
    
    ##add some outliers
    inout = np.where((t>0.2))[0][0:10]
    outl = np.zeros(len(t),float)
    outl[inout] = [0.03,0.1,0.05,0.04,0.03,0.02,0.02,0.01,0.001,0.001]
    inout = inout[0:7]
    ##add a transit
    tr = np.ones(len(t),float)
    dp=0.01
    intr = np.where((t>0.5-2./24.) & (t<0.5+2./24.))[0]
    if ontr == False: 
        intr = np.where((t>0.75-2./24.) & (t<0.75+2./24.))[0]
        inout = np.concatenate((inout,intr[2:-2]))
    tr[intr] = 1.0 - dp
    #pdb.set_trace()
    lc_All = lc*tr + outl
    
    mskarr = np.ones(len(t))
    mskarr[inout] = -1
    qwe = np.where(mskarr > 0)[0]
    
    fitpoly = np.polyval(np.polyfit(t[qwe],lc[qwe]*tr[qwe],2),t)
    trmod = polymod*tr
    if ontr == False: trmod = polymod
    tpoint = np.argmin(np.absolute(np.median(t)-t))
    return t,lc,lc_All,fitpoly,trmod,tpoint,inout

def bulk_run(tic_list,download_dir):
    '''
    Function that does a "bulk run" of the notch pipeline on a provided list of targets.
    Inputs: 1) a list of TIC ID's (tic_list) of the targets of interest in the form of either 
    a Python list or a NumPy array, 2) a download directory path to the folder location in
    which the results will be stored as pickled thymeNL interface objects with accompanying
    notch attributes.
    To be added:
        -ability to choose TESS or K2 mission (adding epic_list instead of TIC LIST)
        -ability to specify SAP or PDCSAP flux
        -Causal Pixel Model (CPM) Functionality implemented with tess_cpm github repository 
    '''
    for i,tic in enumerate(tic_list):
        print("Working on object " + str(i+1) + "/" + str(len(tic_list)) + ".")
        query_string = 'tic ' + str(tic)
        target_name = 'tic' + str(tic)

        lcf = search_lightcurvefile(target_name,mission = 'TESS').download()

        lc_pdcsap = lcf.PDCSAP_FLUX.remove_nans().remove_outliers()
        time = lc_pdcsap.time
        flux = lc_pdcsap.flux
        norm_flux = flux / np.median(flux)
        
        target = thymeNL.target(target_name)
        target.load_data(time,flux)
        target.run_notch(window=0.5,mindbic=-1.0)
        
        with open(os.path.join(download_dir,target_name + '.pkl'),'wb') as outfile:
            pkl.dump(target,outfile)   


