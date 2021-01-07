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
    detected_it,detp,dett0,detdp = core.do_detrend(-1,-1,arclength=False, raw=False, wsize=windowsize,saveoutput=False,resolvabletrans=False,k2sff=False,indata=udata,period_matching = [per,exp_depth,t0],demode=demode,deltabic=usedeltabic,min_period=min_period,max_period = max_period,alias_num=alias_num,tess=True)
    
    
    return detected_it,detp,dett0,detdp




    
    
# def injrec_tess_test(epic,rawdata,per,rp,impact,t0,ecc,omega,mstar,rstar,idstring='',wsize=1.0,demode=1,forcenull=False,alias_num=2.0,min_period=1.0001,max_period=12.0):
#     
#     ##rawdata should be the input data after K2SFF processing.
#     
#     urawdata     = rawdata.copy() ##duplicate it so that it doesn't get overwritten by the injection
#     
#     #find any nans in time
#     qwe      = np.where(np.isnan(urawdata.t))[0]
#     ##The planet signal:
#     injflux  = construct_planet_signal(urawdata.t,per,rp,impact,t0,ecc,omega,mstar,rstar,oversample=20,exposure_time=scipy.stats.mode(urawdata.t-np.roll(urawdata.t,1))[0][0]*24.0*60.0)
#     injflux[qwe] = 1.0
#     ##figure out the expected depth from the batman lightcurve, important when impact parameter is closer to 1.0
#     exp_depth    = 1.0-np.min(injflux)
#     
#    
#     ##now inject the planet signal into the rawdata aperture photometry, this happens to be easy
#     urawdata.fcor *= injflux
#     urawdata.fraw *= injflux
# 
#     ##now run our notch/LCR detrending with the period matching keyword set 
#     if forcenull == False: usedeltabic=-1.0
#     if forcenull == True : usedeltabic = np.inf
#     
#     ##these outputs are a 1/0 flag, the period, and the t0
#     detected_it,detp,dett0,detdp = do_detrend(-1,epic,arclength=False, raw=False, wsize=wsize,saveoutput=False,resolvabletrans=False,k2sff=False,idstring=idstring,indata=urawdata,period_matching = [per,exp_depth,t0],demode=demode,deltabic=usedeltabic,min_period=min_period,max_period = max_period,alias_num=alias_num,tess=True)
#     #print 'done1'
# #    if exp_depth <= 0.0001:
# #        if detected_it == True: print 'Strange!!!!,'+idstring
# #    
#  #   import pdb
# #    pdb.set_trace()
#     return detected_it,detp,dett0,detdp
    