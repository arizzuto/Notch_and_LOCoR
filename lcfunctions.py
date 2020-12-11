import numpy as np
import core
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
    ##combine multiple sectors of data:
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
    
    
    
def run_notch(data,window=0.5,mindbic=-1.0):
    '''
        a wrapper that runs the Notch filter detrending via core.do_detrend()
    '''
    
    print('Running notch filter pipeline with windowsize of ' + str(window) + ' and minimum DeltaBIC of ' +str(mindbic))    
    fittimes,depth,detrend,polyshape,badflag = core.do_detrend(1,1001,arclength=False,raw=False,wsize=window,indata=data,saveoutput=False,outdir='',resolvabletrans=False,demode=1,cleanup=True,deltabic=mindbic)
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
    
def run_locor(data,prot,alias_num=0.01):
    '''
        a wrapper that runs the LOCoR detrending via core.do_detrend()
    '''
    fittimes,depth,detrend,polyshape,badflag = core.do_detrend(1,1001,arclength=False,raw=False,wsize=prot,indata=data,saveoutput=False,outdir='',resolvabletrans=False,demode=2,alias_num=alias_num,cleanup=True)
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
