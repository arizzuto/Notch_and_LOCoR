##standard imports
import numpy as np
import pickle
import scipy.stats
import os

from scipy.optimize import curve_fit

##installed things
from tqdm import tqdm
import mpyfit ##using mpyfit for the sliding window because its a C version and is incredibly fast compared to the python only version. Results are identical.
# import batman ## Kreidbergs BATMAN package

np.set_printoptions(suppress=True)

def do_detrend(cnum, epic, arclength=False, raw=False, wsize=1.0,
               totalfilename='', data_dir='/Volumes/UT2/UTRAID/K2raw/',
               outdir='', saveoutput=True, resolvabletrans=False, k2sff=False,
               indata=np.array([False, False]), idstring='', known_period=[-1,
               -1, -1], known_period2 = [-1, -1, -1], deltabic=-1.0,
               cleanup=False, period_matching=[-1, -1], snrcut=7.0, demode=1,
               max_period=30.0, min_period=1.00001, alias_num=2.0, tess=False,
               show_progress=False, verbose=False):

    '''
    Wrapper for running the notch filter and LOCoR detrending and BLS searches
    "It does this in a controlled way, including reading/writing data."

    Currently the inputs are not ideal, theres some decluttering to do...

    Inputs:Coming soon

    Optional Inputs:Coming soon, lots

    Outputs: Coming soon, lots of options
    '''

    if type(indata[0]) != type(np.array([False])[0]):
        data = indata.copy() ## the case where we passed the data directly

    ##if we know where the planet is and want to aggressively detrend with a transit mask
    ##make a mask array for use in sliding window
    transmask=[-1, -1]
    if (known_period[0] != -1) & (cleanup == True):
        kphase = calc_phase(data.t, known_period[0], known_period[1])
        keep   = np.where((kphase < known_period[2]) | (kphase > known_period[3]))[0]
        transmask = np.zeros(len(kphase), dtype=int)+99
        transmask[keep] = 0

    ##run the detrending sliding notch fitter, LOCoR, or Something Else???!?
    if verbose: 
        print('Running Detrend')
    if demode == 1:
        fittimes, depth, detrend, polyshape, badflag = (
            sliding_window(
                data, windowsize=wsize, use_arclength=arclength, use_raw=raw,
                deltabic=deltabic, resolvable_trans=resolvabletrans,
                cleanmask=transmask, show_progress=show_progress
            ) ##Notch Filter
        )

    if demode == 2:
        fittimes, depth, detrend, polyshape, badflag = (
            rcomb(
                data, wsize, cleanmask=transmask, aliasnum=alias_num
            ) ##LOCoR
        )

    ##for the cleanup case, don't bother with trying to find transits:
    if cleanup == True:
        return fittimes, depth, detrend, polyshape, badflag

    ##run the bls search, rmsclip should really be =1
    if period_matching[0] < 0:
        best_p, dp, t0, detsig, firstpower, pgrid, firstphase, dcyc = (
            bls_transit_search(data, detrend, badflag, rmsclip=1.5,
                               snrcut=snrcut, period_matching=period_matching,
                               searchmax=max_period, searchmin=min_period)
        )
        ## now clean out high-arclength points in each division, then retry

        divs = np.unique(data.divisions).astype(int)
        if tess == False:
            runmask = np.zeros(len(data.t), int)
            for dd in range(len(divs)):
                thisdiv = np.where(data.divisions == divs[dd])[0]
                ##kill 5 % on each side
                keep = np.where(
                    (data.s[thisdiv] < np.percentile(data.s[thisdiv], 96)) &
                    (data.s[thisdiv] > np.percentile(data.s[thisdiv], 4))
                )[0]
                runmask[thisdiv[keep]] = 1

            gomask = np.where(runmask == 1)[0]

        if (len(divs) == 1) | (tess == True):
            print( 'skipping gomask search')
        ##now rerun BLS with these good points only
        if (len(divs) > 1) & (tess == False):
            best_p2, dp2, t02, detsig2, firstpower, pgrid2, firstphase, dcyc2 = (
                bls_transit_search(data[gomask], detrend[gomask],
                                   badflag[gomask], rmsclip=1.5, snrcut=snrcut,
                                   period_matching=period_matching,
                                   searchmax=max_period, searchmin=min_period)
            )

            best_p = np.concatenate((best_p2, best_p))
            dp = np.concatenate((dp2, dp))
            t0 = np.concatenate((t02, t0))
            detsig = np.concatenate((detsig2, detsig))
            dcyc = np.concatenate((dcyc2, dcyc))


        ##for the demode=2 case, run again with all detrend outliers clipped
        if (demode == 2) | (demode == 3):
            gomask = np.where(badflag == 1)[0]
            best_p3, dp3, t03, detsig3, firstpower3, pgrid3, firstphase, dcyc3 = (
                    bls_transit_search(data[gomask], detrend[gomask],
                                       badflag[gomask], rmsclip=1.5,
                                       snrcut=snrcut,
                                       period_matching=period_matching,
                                       searchmax=max_period,
                                       searchmin=min_period)
            )
            best_p = np.concatenate((best_p, best_p3))
            dp = np.concatenate((dp, dp3))
            t0 = np.concatenate((t0, t03))
            detsig = np.concatenate((detsig, detsig3))
            dcyc = np.concatenate((dcyc, dcyc3))


    ##injection recovery input matching section here:
    if period_matching[0] > 0:
        pmatch_result =  (
            bls_transit_search(data, detrend, badflag, rmsclip=1.0,
                               snrcut=snrcut, period_matching=period_matching,
                               searchmax=max_period, searchmin=min_period)
        )
        #now run additional searches if the standard one fails to get you your planet back
        if pmatch_result[0] == 0:

            if (tess == True) & ((demode == 1) | (demode == 5)): ##run the deltabic search, only for notch pipe
                print('Doing BIC Mode')
                bicstat = depth[1]-np.median(depth[1])
                bicstat = 1- bicstat/np.max(bicstat)
                pmatch_result =  (
                    bls_transit_search(data, bicstat, badflag, rmsclip=1.0,
                                       snrcut=snrcut,
                                       period_matching=period_matching,
                                       searchmax=max_period,
                                       searchmin=min_period, datamode='bic')
                )

            ##now clean out high-arclength points in each division, then retry
            divs = np.unique(data.divisions).astype(int)
            if tess == False:
                divs = np.unique(data.divisions)
                runmask = np.zeros(len(data.t), int)
                for dd in range(len(divs)):
                    thisdiv = np.where(data.divisions == divs[dd])[0]
                    ##kill 4% on each side
                    keep = (
                        np.where((data.s[thisdiv] <
                                  np.percentile(data.s[thisdiv], 96)) &
                                 (data.s[thisdiv] >
                                  np.percentile(data.s[thisdiv], 4)))[0]
                    )
                    runmask[thisdiv[keep]] = 1
                ##now run with the new arclength mask for each division

                gomask = np.where(runmask == 1)[0]

            if len(divs) == 1 | (tess == True): print( 'skipping gomask search')
           ## print tess
            if (len(divs) > 1) & (tess == False):
                pmatch_result =  (
                    bls_transit_search(data[gomask], detrend[gomask],
                                       badflag[gomask], rmsclip=1.0,
                                       snrcut=snrcut,
                                       period_matching=period_matching,
                                       searchmax=max_period,
                                       searchmin=min_period)
                )

            ##for demode=2 case, run again with all detrend outliers clipped, if still not finding injected signal
            if (pmatch_result[0] == 0) & (demode == 2): ##for tess and not tess
                gomask = np.where(badflag==1)[0]
                pmatch_result =  (
                    bls_transit_search(data[gomask], detrend[gomask],
                                       badflag[gomask], rmsclip=1.0,
                                       snrcut=snrcut,
                                       period_matching=period_matching,
                                       searchmax=max_period,
                                       searchmin=min_period)
                )

        return pmatch_result

    ##if not saving the output, return useful variables
    if saveoutput == False:
        return (data, fittimes, depth, detrend, polyshape, badflag, pgrid,
                firstpower, firstphase, detsig, best_p, dp, t0, dcyc)

    ##start outputting here, if instructed to, this makes outputfiles and lots of plots
    if saveoutput == True:
        mmm = np.ones(len(data.t), dtype=int)
        bad = np.where(((data.s<0.0) | (data.s>8)) & (detrend<0.99) | (detrend <=0.0))[0]
        mmm[bad] = 0

        ##remove other high points, they can hid transits from BLS
        lcrms = np.sqrt(np.nanmean((1.0-detrend)**2))
        good = np.where((badflag < 2)  & (mmm == 1) & (detrend < 2**lcrms+1.0))[0]
        import matplotlib.pyplot as plt
        savefile = outdir+'detrend_EPIC'+str(epic)+'.pkl'
        if tess == True: savefile = outdir+'detrend_TIC'+str(epic)+'_'+str(cnum)+'.pkl'

        pickle.dump((data, fittimes, depth, detrend, polyshape, badflag, pgrid, firstpower, firstphase, detsig, best_p, dp, t0, dcyc), open(savefile, 'wb'))
        ##save the detrended LC regardless of the presence of a detection:
        #unphased plot
        plt.plot(data.t[good]-data.t[0], detrend[good], '.')
        ax = plt.subplot(111)
        ax.set_ylim([np.max([0.98, np.min(detrend[good])-0.001]), np.min([np.max(detrend[good])+0.001, 1.01])])
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Relative Brightness')
        xlab = ax.xaxis.get_label()
        ylab = ax.yaxis.get_label()
        xlab.set_weight('bold')
        xlab.set_size(12)
        ylab.set_weight('bold')
        ylab.set_size(12)
        [i.set_linewidth(2) for i in ax.spines.itervalues()]
        ax.xaxis.set_tick_params(width=2, labelsize=10)
        ax.yaxis.set_tick_params(width=2, labelsize=10)
        if tess == False: plt.savefig(figdir + 'detrend_EPIC'+str(epic)+'.pdf')
        if tess == True: plt.savefig(figdir + 'detrend_TIC' + str(epic) +'_'+str(cnum)+'.pdf')
        plt.clf()

        ##if a reasonable planet detection is found
        ##phased plot
        if (detsig[0] > snrcut):
            for cnt in range(len(detsig)):
                if detsig[cnt] >= 7.0:
                    detection = 1
                    thisphase = calc_phase(data.t, best_p[cnt], t0[cnt])
                    plt.plot(thisphase[good], detrend[good], '.')
                    ax = plt.subplot(111)
                    ax.set_ylim([np.max([0.98, np.min(detrend[good])-0.001]), np.min([np.max(detrend[good])+0.001, 1.01])])
                    ax.set_xlabel('Phase (P=' + str(np.round(best_p[cnt], decimals=4)) + ' days, ' +str(np.round(detsig[cnt], decimals=1)) + '-sigma)')
                    ax.set_ylabel('Relative Brightness')
                    xlab = ax.xaxis.get_label()
                    ylab = ax.yaxis.get_label()
                    xlab.set_weight('bold')
                    xlab.set_size(12)
                    ylab.set_weight('bold')
                    ylab.set_size(12)
                    [i.set_linewidth(2) for i in ax.spines.itervalues()]
                    ax.xaxis.set_tick_params(width=2, labelsize=10)
                    ax.yaxis.set_tick_params(width=2, labelsize=10)
                    if tess == False: plt.savefig(figdir+'detections/' + 'phase_EPIC'+str(epic)+'_'+str(cnt)+'.pdf')
                    if tess == True: plt.savefig(figdir+'detections/' + 'phase_TIC'+str(epic)+'_'+str(cnum)+'_'+str(cnt)+'.pdf')
                    plt.clf()

        ##if saving don't return anything other than the filename where the results where saved

        return savefile, detection




##a function that bins a lightcurve into a certain number of bins
def binlc(x, y, nbins=100):
    binedges = (np.max(x)-np.min(x))/nbins*np.linspace(0, nbins, nbins+1)+np.min(x)
    binsize  = binedges[1]-binedges[0]
    xbin     = binedges[1:]-binsize/2.0
    meds     = np.zeros(nbins)
    for i in range(nbins):
        inthis  = np.where((x >= binedges[i]) & (x < binedges[i+1]))[0]
        meds[i] = np.nanmedian(y[inthis])

    return xbin, meds


def lcbin(time2, lc, nbins, usemean=False, userobustmean=False, linfit=False):

    time = time2-np.nanmin(time2)

    usemedian = (1-usemean) & (1-userobustmean)

    binsize = (np.nanmax(time)-np.nanmin(time))/float(nbins)
    fbin    = np.empty(nbins)*np.nan
    ebin    = np.empty(nbins)*np.nan
    tbin    = (np.arange(nbins, dtype=float)+0.5)*binsize + np.nanmin(time2)
    if  (userobustmean == True) or (linfit==True): allgoodind = np.array([-1])

    from notch_and_locor.lcfunctions import robustmean
    for i in range(nbins):
        w = np.where((time >= i*binsize) & (time < (i+1)*binsize))[0]
        if len(w) > 0:
            if usemedian == True:
                fbin[i] = np.nanmedian(lc[w])
                if len(w) > 1: ebin[i] = 1.253*np.std(lc[w])/np.sqrt(len(w))

            if usemean == True:
                fbin[i] = np.nanmean(lc[w])
                if len(w) > 1: ebin[i] = np.std(lc[w])/np.sqrt(len(w))

            if userobustmean == True:
                fbin[i], j1, j2, goodind = robustmean(lc[w], 3)
                if len(w) > 1:
                    if len(goodind) > 1:
                        ebin[i] = np.std(lc[w[goodind]])/np.sqrt(len(w))
                    else: ebin[i] = np.inf
                allgoodind = np.concatenate((allgoodind, w[goodind]))

            if linfit == True:
                if len(w) > 2:
                    bg, rms, goodind, polyfit = robustpolyfit(time2[w], lc[w], 1)
                    fbin[i] = np.polyval(polyc, tbin[i])
                    ebin[i] = rms/np.sqrt(len(w))
                    allgoodind = np.concatenate((allgoodind, w[goodind]))
                if len(w) <= 1: ##if not enough points revert to robust mean
                    fbin[i], j1, j2, goodind = robustmean(lc[w], 3, goodind)
                    if len(w) >1:  ##this doesnt make sense!!!!
                        ebin[i] = np.std(lc[w[goodind]])/np.sqrt(len(w))
                        allgoodind = np.concatenate((allgoodind, w[goodind]))


    if robustmean == True: allgoodind = allgoodind[1:]
    return fbin, binsize, ebin, allgoodind, tbin

##function to calculate phases given a period and a starting time for the oscillation
def calc_phase(time, period, p0):
    anchor = np.floor((time-p0)/period)
    phase = (time-p0)/period - anchor
    return phase


##transit like box function. Depth is a fractional depth, width is in days (as is x)
def notchbox(x, depth, width, middle):
    tbox         = x.copy()*0.0 +1
    inside       = np.where(np.absolute(x-middle) < width/2)[0]
    tbox[inside] = 1.0-depth
    return tbox


def transit_window_slide(p, t=None, fl=None, sig_fl=None, s=None, ttime=None, fjac=None, model=False):
    ##if called by mpfit, one-line it all for speed
    #if model == False: return [0, (fl-(p[0] + p[1]*t + p[2]*t**2 + p[3]*s + p[4]*s**2 + p[7]*s**3)*notchbox(t, p[5], p[6], ttime))/sig_fl]
    ##if user wants all the things, do it step by step
    #if model == True :

    polypolyshape = p[0] + p[1]*t + p[2]*t**2 + p[3]*s + p[4]*s**2 + p[7]*s**3


        ##notch box
    transitshape = notchbox(t, p[5], p[6], ttime)
        ##final model
    themod = polyshape*transitshape
        ##leastsq residuals
    resid = (fl-themod)/sig_fl
    if model == True: return themod, polyshape, transitshape, resid
    return [0, resid]

##transit window function for mpyfit usage
def transit_window_slide_pyfit(p, args, model=False):
    ##unpack the arguaments
    t, fl, sig_fl, s, ttime = args

    polyshape = p[0] + p[1]*t + p[2]*t**2 + p[3]*s + p[4]*s**2 + p[7]*s**3



    transitshape = notchbox(t, p[5], p[6], ttime)
    themod = polyshape*transitshape
    resid = (fl-themod)/sig_fl
    if model == True: return themod, polyshape, transitshape, resid
    return resid

##transit like box function. Depth is a fractional depth, width is in days (as is x)
def slopebox(x, depth, width, middle, ierat):
    ietime = ierat*width/2.0
    ##note that ietime is in minutes
    tbox         = x*0.0 +1
    inside       = np.where(np.absolute(x-middle) < width/2)[0]
    tbox[inside] = 1.0-depth ##this makes a box of width length
    ##now slope the ingress and egress
    ing = np.where(x[inside]-middle + width/2.0 < ietime)[0] ##ingress
    egr = np.where(x[inside]-middle -width/2.0 > -ietime)[0] ##ingress

    ##interpolation is slow, should probably define lines
    if len(ing) > 1: tbox[inside[ing]] = -depth/ietime*x[inside[ing]]+depth/ietime*x[inside[0]] + 1.0
    if len(ing) == 1: tbox[inside[ing]] = (1-depth)/2.0+0.5
    if len(egr) >1 : tbox[inside[egr]] = depth/ietime*x[inside[egr]] +1.0-depth/ietime*x[inside[-1]]
    if len(egr) == 1: tbox[inside[egr]] = (1-depth)/2.0+0.5

    #import pdb
    #import matplotlib.pyplot as plt
    #pdb.set_trace()

    return tbox


def transit_window4_slide_pyfit(p, args, model=False):
    ##unpack the arguaments
    t, fl, sig_fl, s, ttime = args
    #import pdb
    #pdb.set_trace()

    polyshape = p[0] + p[1]*t + p[2]*t**2

    transitshape = slopebox(t, p[3], p[4], ttime, p[5])
    themod = polyshape*transitshape
    resid = (fl-themod)/sig_fl
    if model == True: return themod, polyshape, transitshape, resid
    return resid


def sliding_window(data, windowsize=0.5, use_arclength=False, use_raw=False,
                   efrac=1e-3, resolvable_trans=False, cleanmask=[-1, -1],
                   deltabic=-1.0, animator=False, animatorfunc=None,
                   show_progress=True):
    '''
    Sliding Window Notch-Filter

    This code takes a lightcurve and applies the notch-filter method to remove
    rotation and preserve trasits.

    Inputs:
    (1) data: The data recarray that this code uses for passing data around

    Optional Inputs:

    (1) windowsize: Detrending window size in days, default is 0.5 days.

    (2) use_arclength: Set to True to do a full fit over time and arclength,
    default is False

    (3) use_raw: Set to True to use raw data that is uncorrected for K2
    pointing systematics. Default is False

    (4) efrac: starting fractional uncertainty on the lightcurve data. Default
    is 1mmag. This value is dynamically determined in each fitting window
    and so should only be change if extreme circumstances.

    (4) resolvable_trans: Set to not use the 45 min transit window trail.
    Default is False.

    (5) cleanmask: binary mask to remove a set of points from the fitting.
    Great for masking over a transit signal that you dont want influencing the
    fit. Default is [-1, -1] which turns it off

    (6) deltabic: Bayesian information cirterion difference between the transit
    and no-transit model required to select the transit model. A higher value
    indicates more required evidence. Default is -1.0, which is at least equal
    evidence with a ~1 margin for uncertainty. Set to np.inf to always choose
    the null model or -np.inf to always choose the transit model.

    Outputs:
    (1) Times: The input time axis from data

    (2) depths: notch filter depths at each point in the lightcurve. zero when
    null model chosen

    (3) detrend: detrended lightcurve

    (4) polyshape: the model used to detrend the input lightcurve

    (5) badflag: integer flags for each datapoint in data with 0=fine 1=masked
    as outlier in iterations at least once but still fine, 2=strong positive
    outlier or other suspect point.
    '''

    import time
    wsize  = windowsize
    wnum   = windowsize/(data.t[10]-data.t[9])
    fittimes = data.t
    depthstore = fittimes.copy()*0.0
    dbic       = fittimes.copy()*0.0
    detrend    = data.fraw*0.0
    polyshape  = data.fraw*0.0
    badflag    = np.zeros(len(data.fraw), int)
    badflag2   = badflag.copy()
    running_std_depth = 0.0
    cliplim = 3.5
    if cleanmask[0] != -1: cliplim=4.5
    ##sliding storage for detrending shape not using for now
    #slidestore = np.zeros((len(data.t), wnum*2))*np.nan
    #storepos   = np.zeros(len(data.t), int)
    start=0

    if animator == True:
        start = animatorfunc

    if show_progress == True:
        itit = tqdm(range(start, len(fittimes)))
    else:
        itit = range(start, len(fittimes))

    for i in itit:

        # grab the window
        wind     = np.where((data.t < fittimes[i]+wsize/2.0) & (data.t > fittimes[i]-wsize/2.0))[0]
        wdat     = data[wind].copy()
        starttime = wdat.t[0]*1.0
        thistime = fittimes[i]-wdat.t[0]
        wdat.t  -= wdat.t[0]
        ttt      = np.where(wdat.t == thistime)[0]
        if cleanmask[0] != -1:
            wcleanmask = cleanmask[wind].copy()

        # switch out the raw flux for the Vanderburg flat-fielded flux if not
        # using arc-length parameters
        if use_raw == False:
            wdat.fraw = wdat.fcor

        # if you have only one point in this window, don't try fitting anything
        # (because np.polyfit will yield a LinAlg error).  assign defaults, and
        # skip to the next window.
        if len(wdat) == 1:

            modpoly = np.array([wdat.fraw])
            pos_outlier = np.array([], dtype=int)
            flare = np.array([], dtype=int)
            pars = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            ttt           = np.where(wdat.t == thistime)[0]
            detrend[i]    = wdat.fraw[ttt]/modpoly[ttt]
            polyshape[i]  = modpoly[ttt]*1.0
            depthstore[i] = pars[5]*1.0
            badflag[wind[pos_outlier]] = 1
            badflag[wind[flare]] = 2
            if cleanmask[0] != -1:
                intrans = np.where(cleanmask == 99)[0]
                badflag[intrans] = 0

            if show_progress:
                print(f"For index {i}, caught single-point window. "
                      "Skipping to end!")

            continue

        # now linearize the data for initial checking:
        # and impose a threshold cut for flaring events
        line      = np.polyval(np.polyfit(wdat.t, wdat.fraw, 1), wdat.t)
        lineoff   = np.sqrt(np.nanmean((wdat.fraw-line)**2))
        lineresid = (wdat.fraw-line)/lineoff
        flare  = np.where((lineresid > 8.0) | (wdat.fraw < 0.0))[0]
        wdat.qual[flare] = 2

        # fit the model to the window at this transit time
        # build the inputs for MPYFIT
        pstart  = np.array([np.nanmedian(wdat.fraw), -0.01, -0.01, 0.0, 0.0, 0.002, 0.08, 0.0])
        # run a polyfit to get starting parameters for a full fit:
        initpoly = np.polyfit(wdat.t, wdat.fraw, 2)
        pstart[0] = initpoly[2]
        pstart[1]  = initpoly[1]
        pstart[2] = initpoly[0]
        error   = efrac*wdat.fraw
        # find zero points, this happens on rare occasions and messes things up badly if not dealt with
        qwe = np.where(wdat.fraw == 0.0)[0]
        error[qwe] = np.inf ##assign it a dramatic error

        ##find a transit in the first third of the window that has already been passed
        ##this is actually not a good idea: at best a duplication of the outlier steps, at worst, 
        ##eats points that should be included in fit
        #passed_trans = np.where((detrend[wind] < 0.999) & (wind-wind[0] < len(wind)/3))[0]
        ##what we really should do is keep a running depth standard deviation

        passed_trans = np.where(
            (depthstore[wind] > 5.0*np.std(depthstore[0:i+1])) & (wind-wind[0] < len(wind)/3)
        )[0]

        if i < 10:
            passed_trans = np.where(depthstore[wind] > 1e10)[0]

        # has something previously been flagged as bad? This should only be
        # points ID'd as flares in passed windows this step basically is an
        # insurance policy for when multiple flares creep into a window making
        # clipping hard.
        suspect = np.where((badflag[wind] ==2) & (wind != ttt))[0]

        # combine the flare points, suspect points, and passed_transits
        dontuse = np.unique(np.append(np.append(flare, suspect), passed_trans))

        # if running the cleanup detrending when we known were the transit is:
        # this should mean all in-transit points don't alter the detrending fit
        if cleanmask[0] != -1:
            intrans = np.where(wcleanmask == 99)[0]
            dontuse = np.unique(np.append(dontuse, intrans))

        # set all these points to bad
        error[dontuse] = np.inf

        # set up the fitting parinfo dictionary
        parinfo2 = [{'fixed':False, 'limits':(None, None), 'step':0.1} for dddd in range(pstart.shape[0])]
        parinfo2[5]['step']      = 1.0
        parinfo2[6]['step']      = 0.0105
        parinfo2[5]['limits']    = (0.0, 1.0)
        parinfo2[6]['limits']    = (0.02, 0.207)

        # if arc length is not going to be used in the fit, fix those parameters
        if use_arclength == False:
            parinfo2[3]['fixed'] = True
            parinfo2[4]['fixed'] = True
            parinfo2[7]['fixed'] = True

        # make arclength fit linear when using corrected curves and arclength
        if (use_raw == False) & (use_arclength == True):
            parinfo2[4]['fixed'] = True

        # run on a grid of transit durations for the notch, for things that are planet like in depth
        lgrid = np.array([0.75, 1.0, 2.0, 4.0])/24.0
        numpars = len(parinfo2) - np.sum([parinfo2[dude]['fixed'] for dude in range(len(parinfo2))])

        # if hardly any points in window, just use the initial fit
        if len(wdat) < numpars+1+5:
            modpoly = np.polyval(initpoly, wdat.t)
            pos_outlier = np.array([], dtype=int)
            pars = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # if enough points in this window to do the full notch filter fit, go fit it
        else:
            # if switch to set, don't try the single point transit (45 mins)
            # also adjust related fit limit
            if resolvable_trans == True:
                lgrid = np.array([1.0, 2.0, 4.0])/24.0
                parinfo2[6]['limits'] = (0.0415, 0.2)
            # fix the duration in the grid fitting
            parinfo2[6]['fixed'] = True
            bestpars             = pstart*0.0
            current_chi2         = np.inf # not going to store anything except the current best solution on the grid

            for l in range(len(lgrid)):

                pstart[6] = lgrid[l]

                ##args = t, fl, sig_fl, s, ttime
                dfit2, extrares = mpyfit.fit(
                    transit_window_slide_pyfit, pstart,
                    args=(wdat.t, wdat.fraw, error, wdat.s, thistime),
                    parinfo=parinfo2, maxiter=200
                )

                if extrares['bestnorm'] < current_chi2:
                    bestpars     = dfit2*1.0
                    current_chi2 = extrares['bestnorm']*1.0

            # Now that best grid solution is found, unfix the transit duration paramater
            parinfo2[6]['fixed'] =False

            # run the fit with no fixed parameters but from the best grid point result from above
            pstart  = bestpars
            pars, extrares = mpyfit.fit(
                transit_window_slide_pyfit, pstart,
                args=(wdat.t, wdat.fraw, error, wdat.s, thistime),
                parinfo=parinfo2, maxiter=200
            )

            themod, modpoly, modnotch, modresid = (
                transit_window_slide_pyfit(
                    pars, args=(wdat.t, wdat.fraw, error, wdat.s, thistime),
                    model=True)
            )

            # remove high outliers (like flares) and run again, do this based
            # on a statistical measure like rms loop a few times to converge on
            # an outlier-less solution run the outlier rejection 5 times, that
            # should be enough....
            pos_outlier = np.zeros(0, int)
            cliperr     = error*1.0
            oldnum=0

            for lll in range(0, 5):

                koff     = np.where(np.isinf(cliperr) == False)[0]
                rms      = np.sqrt(np.mean((wdat.fraw[koff]-themod[koff])**2))
                rmsoff   = (wdat.fraw - themod)/rms
                knownbad = np.where(np.isinf(cliperr))[0]

                if np.isnan(rms) | (rms ==0):
                    break

                # identify outliers
                new_outl    = np.where((rmsoff > cliplim) | (rmsoff < -cliplim))[0]
                pos_outlier = np.unique(np.append(pos_outlier, new_outl))
                cliperr[:]  = rms*1.0
                cliperr[pos_outlier] = np.inf   # set outlier point uncertainties to effective infinity
                cliperr[knownbad]    = np.inf   # set outlier point uncertainties to effective infinity

                # dont refit unless theres a new point flagged as an outlier.
                if (len(pos_outlier) > oldnum) | (lll==0):

                    fa      = {'fl':wdat.fraw, 'sig_fl':cliperr, 't':wdat.t,
                               's':wdat.s, 'ttime':thistime}

                    args = (wdat.t, wdat.fraw, error, wdat.s, thistime)
                    pars, thisfit = mpyfit.fit(
                        transit_window_slide_pyfit, pstart,
                        args=(wdat.t, wdat.fraw, cliperr, wdat.s, thistime),
                        parinfo=parinfo2
                    )

                    themod, modpoly, modnotch, modresid = (
                        transit_window_slide_pyfit(
                            pars, args=(
                                wdat.t, wdat.fraw, cliperr, wdat.s, thistime
                            ), model=True)
                    )

                oldnum = len(pos_outlier)

            # calculate the final model chi2 here, need to recalc because the
            # rms number might have changed after the last outlier rejection
            # and model fit

            modchi2 = np.sum(((themod-wdat.fraw)/cliperr)**2)

             #try model with no transit, i.e fix the depth to zero
            parinfo2[5]['fixed'] = True
            nullstart           = pars.copy()
            nullstart[5]        = 0.0

            nullfit, nullextra = mpyfit.fit(
                transit_window_slide_pyfit, nullstart,
                args=(wdat.t, wdat.fraw, cliperr, wdat.s, thistime),
                parinfo=parinfo2
            )

            nullmod, nullpoly, dummy, nullresid = (
                transit_window_slide_pyfit(
                    nullfit, args=(
                        wdat.t, wdat.fraw, cliperr, wdat.s, thistime
                    ), model=True)
            )

            nullchi2 = np.sum(nullresid**2)

            # what are the two model bayesian information criteria?
            numpars = len(parinfo2) - np.sum([parinfo2[dude]['fixed'] for dude in range(len(parinfo2))])
            npoints  = len(np.where(cliperr < 0.99)[0])
            bicnull  = nullchi2 + (numpars-2)*np.log(npoints)
            bicfull  = modchi2  + numpars*np.log(npoints)

            if animator == True:
                # here take the axes passed in an plot the latest stuff for the
                # animator then immediately return
                print('animating')
                return (
                    wdat.t, themod, wdat.fraw, nullpoly, pos_outlier, dontuse,
                    bicnull - bicfull
                )

            # if the bayesian information criterion for the full and null
            # models provide evidence against the transit notch, just use the
            # null model for the fit, the difference required to believe a
            # transit notch is required is by default 0, which is very
            # inclusive, but can be tuned if you know what you're looking for.
            # deltaBIC > 0 means notch is favoured.
            dbic[i] = bicnull - bicfull
            if dbic[i] < deltabic:
                modpoly = nullmod # if notch not required, use the null model
                ##pars[5] = 0.0     ##set to zero for later storage, actually retain it for record keeping
            if  i > np.inf: ##for Aaron to look at whats getting notched, set i>-1 to see everything
                print( fittimes[i])
                print( bicnull-bicfull)
                #print pars[5]
                #print np.min(cliperr)
                #print rms
                import pdb
                import matplotlib.pyplot as plt
                plt.plot(wdat.t, wdat.fraw, '.')
                plt.errorbar(wdat.t, wdat.fraw, yerr = cliplim*0.0+rms, fmt='b.')
                plt.plot(wdat.t[dontuse], wdat.fraw[dontuse], 'ko')
                plt.plot(wdat.t, themod, 'g')
                plt.plot(wdat.t, nullmod, 'r')
                plt.plot(wdat.t[ttt], wdat.fraw[ttt], 'm*', markersize=18)
                plt.plot(wdat.t[pos_outlier], wdat.fraw[pos_outlier], 'rx', markeredgecolor='r', mew=2)
                plt.show()
                pdb.set_trace()

        # save things we care about, like a detrended curve and a transit
        # depth fit for this point in time

        ttt           = np.where(wdat.t == thistime)[0]
        detrend[i]    = wdat.fraw[ttt]/modpoly[ttt]
        polyshape[i]  = modpoly[ttt]*1.0
        depthstore[i] = pars[5]*1.0
        ##make sure outliers get flagged appropriately
        badflag[wind[pos_outlier]] = 1
        badflag[wind[flare]] = 2
        if cleanmask[0] != -1:
            intrans = np.where(cleanmask == 99)[0]
            badflag[intrans] = 0

    # spit output
    return fittimes, [depthstore, dbic], detrend, polyshape, badflag


#@profile
def sliding_window4(data, windowsize=0.5, use_raw=False, efrac=1e-3, resolvable_trans=False, cleanmask=[-1, -1], deltabic=-1.0):


    import time
    wsize  = windowsize
    wnum   = windowsize/(data.t[10]-data.t[9])
    fittimes = data.t
    depthstore = fittimes.copy()*0.0
    dbic       = fittimes.copy()*0.0
    detrend    = data.fraw*0.0
    polyshape  = data.fraw*0.0
    badflag    = np.zeros(len(data.fraw), int)
    badflag2   = badflag.copy()
    running_std_depth = 0.0
    cliplim = 3.0
    if cleanmask[0] != -1: cliplim=4.5
    ##sliding storage for detrending shape not using for now
    #slidestore = np.zeros((len(data.t), wnum*2))*np.nan
    #storepos   = np.zeros(len(data.t), int)
    for i in range(0, len(fittimes)):     ##usually len(fittimes)
        print('up to ' + str(i) + ' of ' +str(len(fittimes)))
        ##an upper sco transit point 3000:
        #if np.mod(i+1, 100) == 0: print 'Up to ' + str(i+1) + ' out of ' + str(len(fittimes)) + ' times'

        ##grab the window
        wind     = np.where((data.t < fittimes[i]+wsize/2.0) & (data.t > fittimes[i]-wsize/2.0))[0]
        wdat     = data[wind].copy()
        starttime = wdat.t[0]*1.0
        thistime = fittimes[i]-wdat.t[0]
        wdat.t  -= wdat.t[0]
        ttt      = np.where(wdat.t == thistime)[0]
        if cleanmask[0] != -1: wcleanmask = cleanmask[wind].copy()

        ##switch out the raw flux for the Vanderburg flat-fielded flux if not using arc-length parameters
        if use_raw == False: wdat.fraw = wdat.fcor


        ##now linearize the data for initial checking:
        ##and impose a threshold cut for flaring events
        line      = np.polyval(np.polyfit(wdat.t, wdat.fraw, 1), wdat.t)
        lineoff   = np.sqrt(np.nanmean((wdat.fraw-line)**2))
        lineresid = (wdat.fraw-line)/lineoff
        #mlevel = np.median(wdat.fraw)
        #if use_raw == False: flare  = np.where((wdat.fraw/mlevel > 1.005) | (lineresid > 3.0))[0]
        #if use_raw == True:  flare  = np.where(lineresid > 3.0)[0]
        flare  = np.where((lineresid > 8.0) | (wdat.fraw < 0.0))[0]
        wdat.qual[flare] = 2
        #extrabad = np.where(wdat.fraw < 0.0)[0]
        #flar

        ##fit the model to the window at this transit time
        ##build the inputs for MPYFIT
        pstart  = np.array([np.nanmedian(wdat.fraw), -0.01, -0.01, 0.002, 0.08, 0.1])
        ##run a polyfit to get starting parameters for a full fit:
        initpoly = np.polyfit(wdat.t, wdat.fraw, 2)
        pstart[0] = initpoly[2]
        pstart[1]  = initpoly[1]
        pstart[2] = initpoly[0]
        error   = efrac*wdat.fraw
        ##find zero points, this happens on rare occasions and messes things up badly if not dealt with
        qwe = np.where(wdat.fraw == 0.0)[0]
        error[qwe] = np.inf ##assign it a dramatic error

        ##find a transit in the first third of the window that has already been passed
        ##this is actually not a good idea: at best a duplication of the outlier steps, at worst, 
        ##eats points that should be included in fit
        #passed_trans = np.where((detrend[wind] < 0.999) & (wind-wind[0] < len(wind)/3))[0]
        ##what we really should do is keep a running depth standard deviation
        passed_trans = np.where((depthstore[wind] >=5.0*np.std(depthstore[0:i+1])) & (wind-wind[0] < len(wind)/3))[0]
        if i < 10: passed_trans = np.where(depthstore[wind] > 1e10)[0]

        ##has something previously been flagged as bad? This should only be points ID'd as flares in passed windows
        ##this step basically is an ensurance policy for when multiple flares creep into a window making clipping hard.
        suspect = np.where((badflag[wind] ==2) & (wind != ttt))[0]

        ##combine the flare points, suspect points, and passed_transits
        dontuse = np.unique(np.append(np.append(flare, suspect), passed_trans))
        #import pdb
        #pdb.set_trace()
        ##if running the cleanup detrending when we known were the transit is:
        ##this should mean all in-transit points don't alter the detrending fit
        if cleanmask[0] != -1:
            intrans = np.where(wcleanmask == 99)[0]
            dontuse = np.unique(np.append(dontuse, intrans))

        ##set all these points to bad
        error[dontuse] = np.inf

        ##set up the fitting parinfo dictionary
        parinfo2 = [{'fixed':False, 'limits':(None, None), 'step':0.1} for dddd in range(pstart.shape[0])]
        parinfo2[3]['step']      = 1.0
        parinfo2[4]['step']      = 0.0105
        parinfo2[3]['limits']    = (0.0, 1.0)
        parinfo2[4]['limits']    = (0.02, 0.2)
        parinfo2[5]['limits']    = (0.05, 0.15) ##lets make this t_ie/tfull

        ##run on a grid of transit durations for the notch, for things that are planet like in depth
        lgrid = np.array([0.75, 1.0, 2.0, 4.0])/24.0
        numpars = len(parinfo2) - np.sum([parinfo2[dude]['fixed'] for dude in range(len(parinfo2))])

        if len(wdat) < numpars+1+5:##if hardly any points in window, just use the initial fit
            modpoly = np.polyval(initpoly, wdat.t)
            pos_outlier = np.array([], dtype=int)
            pars = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            #print( 'in nodata fit')
        else: ##if enough points in this window to do the full notch filter fit, go fit it
            #print 'in full fit'


            ##if switch to set, don't try the single point transit (45 mins)
            ##also adjust related fit limit
            if resolvable_trans == True:
                lgrid = np.array([1.0, 2.0, 4.0])/24.0
                parinfo2[4]['limits'] = (0.0415, 0.2)
            ##fix the duration in the grid fitting
            parinfo2[4]['fixed'] = True
            bestpars             = pstart*0.0
            current_chi2         = np.inf ##not going to store anything except the current best solution on the grid
            for l in range(len(lgrid)):
                pstart[4] = lgrid[l]
                #import pdb
                #pdb.set_trace()
                dfit2, extrares = mpyfit.fit(transit_window4_slide_pyfit, pstart, args=(wdat.t, wdat.fraw, error, wdat.s, thistime), parinfo=parinfo2, maxiter=200) ##args = t, fl, sig_fl, s, ttime
                if extrares['bestnorm'] < current_chi2:
                    bestpars     = dfit2*1.0
                    current_chi2 = extrares['bestnorm']*1.0
            ##Now that best grid solution is found, unfix the transit duration paramater
            parinfo2[4]['fixed'] =False

            ##run the fit with no fixed parameters but from the best grid point result from above
            pstart  = bestpars
            pars, extrares = mpyfit.fit(transit_window4_slide_pyfit, pstart, args=(wdat.t, wdat.fraw, error, wdat.s, thistime), parinfo=parinfo2, maxiter=200)
            themod, modpoly, modnotch, modresid = transit_window4_slide_pyfit(pars, args=(wdat.t, wdat.fraw, error, wdat.s, thistime), model=True)

            ##remove high outliers (like flares) and run again, do this based on a statistical measure like rms
            ##loop a few times to converge on an outlier-less solution
            ##run the outlier rejection 5 times, that should be enough....
            pos_outlier = np.zeros(0, int)
            cliperr     = error*1.0
            oldnum=0



            for lll in range(0, 5):
                koff     = np.where(np.isinf(cliperr) == False)[0]
                rms      = np.sqrt(np.mean((wdat.fraw[koff]-themod[koff])**2))
                rmsoff   = (wdat.fraw - themod)/rms
                knownbad = np.where(np.isinf(cliperr))[0]
                if np.isnan(rms) | (rms ==0): break


                ##identify outliers
                new_outl    = np.where((rmsoff > cliplim) | (rmsoff < -cliplim))[0]
                pos_outlier = np.unique(np.append(pos_outlier, new_outl))
                cliperr[:]  = rms*1.0
                cliperr[pos_outlier] = np.inf   ##set outlier point uncertainties to effective infinity
                cliperr[knownbad]    = np.inf   ##set outlier point uncertainties to effective infinity

                if (len(pos_outlier) > oldnum) | (lll==0): ##dont refit unless theres a new point flagged as an outlier.
                    fa      = {'fl':wdat.fraw, 'sig_fl':cliperr, 't':wdat.t, 's':wdat.s, 'ttime':thistime}
                    args=(wdat.t, wdat.fraw, error, wdat.s, thistime)
                    pars, thisfit = mpyfit.fit(transit_window4_slide_pyfit, pstart, args=(wdat.t, wdat.fraw, cliperr, wdat.s, thistime), parinfo=parinfo2)
                    themod, modpoly, modnotch, modresid = transit_window4_slide_pyfit(pars, args=(wdat.t, wdat.fraw, cliperr, wdat.s, thistime), model=True)

                oldnum = len(pos_outlier)
            ##calculate the final model chi2 here, need to recalc because the rms number might have changed after the last outlier rejection and model fit
            modchi2 = np.sum(((themod-wdat.fraw)/cliperr)**2)

            ##try model with no transit, i.e fix the depth to zero
            parinfo2[3]['fixed'] = True
            nullstart           = pars.copy()
            nullstart[3]        = 0.0

            nullfit, nullextra = mpyfit.fit(transit_window4_slide_pyfit, nullstart, args=(wdat.t, wdat.fraw, cliperr, wdat.s, thistime), parinfo=parinfo2)
            nullmod, nullpoly, dummy, nullresid = transit_window4_slide_pyfit(nullfit, args=(wdat.t, wdat.fraw, cliperr, wdat.s, thistime), model=True)
            nullchi2 = np.sum(nullresid**2)

            ##what are the two model bayesian information criteria?
            numpars = len(parinfo2) - np.sum([parinfo2[dude]['fixed'] for dude in range(len(parinfo2))])
            npoints  = len(np.where(cliperr < 0.99)[0])
            bicnull  = nullchi2 + (numpars-2)*np.log(npoints)
            bicfull  = modchi2  + numpars*np.log(npoints)

            ##if the bayesian information criterion for the full and null models provide evidence against the transit notch, 
            ##just use the null model for the fit, the difference required to believe a transit notch is requires is by
            ##default 0, which is very inclusive, but can be tuned if you know what you're looking for.
            ##deltaBIC > 0 means notch is favoured.
            dbic[i] = bicnull - bicfull
            if dbic[i] < deltabic:
                modpoly = nullmod ##if notch not required, use the null model
                ##pars[5] = 0.0     ##set to zero for later storage, actually retain it for record keeping
            if  i > np.inf: ##for Aaron to look at whats getting notched, set i>-1 to see everything
                print( fittimes[i])
                print( bicnull-bicfull)
                #print pars[5]
                #print np.min(cliperr)
                #print rms
                import pdb
                import matplotlib.pyplot as plt
                plt.plot(wdat.t, wdat.fraw, '.')
                plt.errorbar(wdat.t, wdat.fraw, yerr = cliplim*0.0+rms, fmt='b.')
                plt.plot(wdat.t[dontuse], wdat.fraw[dontuse], 'ko')
                plt.plot(wdat.t, themod, 'g')
                plt.plot(wdat.t, nullmod, 'r')
                plt.plot(wdat.t[ttt], wdat.fraw[ttt], 'm*', markersize=18)
                plt.plot(wdat.t[pos_outlier], wdat.fraw[pos_outlier], 'rx', markeredgecolor='r', mew=2)
                plt.show()
                pdb.set_trace()

        ##save things we care about, like a detrended curve and a transit depth fit for this point in time
        ttt           = np.where(wdat.t == thistime)[0]
        detrend[i]    = wdat.fraw[ttt]/modpoly[ttt]
        polyshape[i]  = modpoly[ttt]*1.0
        depthstore[i] = pars[3]*1.0
        ##make sure outliers get flagged appropriately
        badflag[wind[pos_outlier]] = 1
        badflag[wind[flare]] = 2
        if cleanmask[0] != -1:
            intrans = np.where(cleanmask == 99)[0]
            badflag[intrans] = 0

    ##spit output
    return fittimes, [depthstore, dbic], detrend, polyshape, badflag



def spotbreak_model(t, breakpos, breaksize): ##model spot appearance/dissapearance as a hat function
    model = np.ones(len(t))
    #import pdb
    #pdb.set_trace()
    spot = np.where(t>breakpos)[0]
    model[np.where(t>breakpos)[0]]=1+breaksize ##a positive break size means an upwards jump in the curve, negative the opposite
    return model

##transit window function for mpyfit usage DEVELOPMENT VERSION OF NOTCH MODEL
def transit_window_slide_pyfit3(p, args, model=False):
    ##unpack the arguaments
    t, fl, sig_fl, s, ttime = args
    #import pdb
    #pdb.set_trace()

   # import warnings
   # warnings.filterwarnings('error')
    #try:
    polyshape = p[0] + p[1]*t + p[2]*t**2 +p[8]*t**3+ + p[9]*t**4+ p[3]*s + p[4]*s**2 + p[7]*s**3
    #except Warning:
        #import pdb
        #pdb.set_trace()

    transitshape = notchbox(t, p[5], p[6], ttime)
    spotshape = spotbreak_model(t, p[8], p[9])

    themod = polyshape*transitshape*spotshape


    resid = (fl-themod)/sig_fl
    if model == True: return themod, polyshape, transitshape, spotshape, resid
    return resid


##more complex poly version of sliding window DEVELOPMENT VERSION OF NOTCH WINDOWER
def sliding_window3(data, windowsize=0.5, use_arclength=False, use_raw=False, efrac=1e-3, resolvable_trans=False, cleanmask=[-1, -1], deltabic=-1.0):
    '''
    Sliding Window Notch-Filter

    THis code takes a lightcurve and applies the notch-filter method to remove rotation and preserve trasits.

    Inputs:
    (1) data: The data recarray that this code uses for passing data around

    Optional Inputs:
    (1) windowsize: Detrending window size in days, default is 0.5 days.
    (2) use_arclength: Set to True to do a full fit over time and arclength, default is False
    (3) use_raw: Set to True to use raw data that is uncorrected for K2 pointing systematics. Default is False
    (4) efrac: starting fractional uncertainty on the lightcurve data. Default is 1mmag. This value is dynamically determined in each fitting window
    and so should only be change if extreme circumstances.
    (4) resolvable_trans: Set to not use the 45 min transit window trail. Default is False.
    (5) cleanmask: binary mask to remove a set of points from the fitting. Great for masking over a transit signal that you
    dont want influencing the fit. Default is [-1, -1] which turns it off
    (6) deltabic: Bayesian information cirterion difference between the transit and no-transit model required to select the transit model. A higher value
    indicates more required evidence. Default is -1.0, which is at least equal evidence with a ~1 margin for uncertainty. Set to np.inf to always choose the null model
    or -np.inf to always choose the transit model.

    Outputs:
    (1) Times: The input time axis from data
    (2) depths: notch filter depths at each point in the lightcurve. zero when null model chosen
    (3) detrend: detrended lightcurve
    (4) polyshape: the model used to detrend the input lightcurve
    (5) badflag: integer flags for each datapoint in data with 0=fine
        1=masked as outlier in iterations at least once but still fine, 2=strong positive outlier or other suspect point.
    '''





    import time
    wsize  = windowsize
    wnum   = windowsize/(data.t[10]-data.t[9])
    fittimes = data.t
    depthstore = fittimes.copy()*0.0
    dbic       = fittimes.copy()*0.0
    detrend    = data.fraw*0.0
    polyshape  = data.fraw*0.0
    badflag    = np.zeros(len(data.fraw), int)
    badflag2   = badflag.copy()
    running_std_depth = 0.0
    cliplim  = 2. ###lower outlier rejection clip limit
    cliplimu = 2. ###upper outlier rejection clip limit





    if cleanmask[0] != -1: cliplim=4.5
    ##sliding storage for detrending shape not using for now
    #slidestore = np.zeros((len(data.t), wnum*2))*np.nan
    #storepos   = np.zeros(len(data.t), int)
    for i in range(1200, len(fittimes)):
        ##an upper sco transit point 3000:
        #if np.mod(i+1, 100) == 0: print 'Up to ' + str(i+1) + ' out of ' + str(len(fittimes)) + ' times'

        ##grab the window
        wind     = np.where((data.t < fittimes[i]+wsize/2.0) & (data.t > fittimes[i]-wsize/2.0))[0]
        wdat     = data[wind].copy()
        starttime = wdat.t[0]*1.0
        thistime = fittimes[i]-wdat.t[0]
        wdat.t  -= wdat.t[0]
        ttt      = np.where(wdat.t == thistime)[0]
        if cleanmask[0] != -1: wcleanmask = cleanmask[wind].copy()

        ##switch out the raw flux for the Vanderburg flat-fielded flux if not using arc-length parameters
        if use_raw == False: wdat.fraw = wdat.fcor


        ##now linearize the data for initial checking:
        ##and impose a threshold cut for flaring events
        line      = np.polyval(np.polyfit(wdat.t, wdat.fraw, 1), wdat.t)
        lineoff   = np.sqrt(np.nanmean((wdat.fraw-line)**2))
        lineresid = (wdat.fraw-line)/lineoff
        #mlevel = np.median(wdat.fraw)
        #if use_raw == False: flare  = np.where((wdat.fraw/mlevel > 1.005) | (lineresid > 3.0))[0]
        #if use_raw == True:  flare  = np.where(lineresid > 3.0)[0]
        flare  = np.where((lineresid > 8.0) | (wdat.fraw < 0.0))[0]
        wdat.qual[flare] = 2
        #extrabad = np.where(wdat.fraw < 0.0)[0]
        #flar

        ##fit the model to the window at this transit time
        ##build the inputs for MPYFIT
        pstart  = np.array([np.nanmedian(wdat.fraw), -0.01, -0.01, 0.0, 0.0, 0.002, 0.08, 0.0, windowsize/2, 0.000])
        ##run a polyfit to get starting parameters for a full fit:
        initpoly = np.polyfit(wdat.t, wdat.fraw, 2)
        pstart[0] = initpoly[2]
        pstart[1]  = initpoly[1]
        pstart[2] = initpoly[0]

        error   = efrac*wdat.fraw
        ##find zero points, this happens on rare occasions and messes things up badly if not dealt with
        qwe = np.where(wdat.fraw == 0.0)[0]
        error[qwe] = np.inf ##assign it a dramatic error

        ##find a transit in the first third of the window that has already been passed
        ##this is actually not a good idea: at best a duplication of the outlier steps, at worst, 
        ##eats points that should be included in fit
        #passed_trans = np.where((detrend[wind] < 0.999) & (wind-wind[0] < len(wind)/3))[0]
        ##what we really should do is keep a running depth standard deviation
        passed_trans = np.where((depthstore[wind] >=5.0*np.std(depthstore[0:i+1])) & (wind-wind[0] < len(wind)/3))[0]
        if i < 10: passed_trans = np.where(depthstore[wind] > 1e10)[0]

        ##has something previously been flagged as bad? This should only be points ID'd as flares in passed windows
        ##this step basically is an ensurance policy for when multiple flares creep into a window making clipping hard.
        suspect = np.where((badflag[wind] ==2) & (wind != ttt))[0]

        ##combine the flare points, suspect points, and passed_transits
        dontuse = np.unique(np.append(np.append(flare, suspect), passed_trans))

        ##if running the cleanup detrending when we known were the transit is:
        ##this should mean all in-transit points don't alter the detrending fit
        if cleanmask[0] != -1:
            intrans = np.where(wcleanmask == 99)[0]
            dontuse = np.unique(np.append(dontuse, intrans))

        ##set all these points to bad
        error[dontuse] = np.inf

        ##set up the fitting parinfo dictionary
        parinfo2 = [{'fixed':False, 'limits':(None, None), 'step':0.1} for dddd in range(pstart.shape[0])]
        parinfo2[5]['step']      = 1.0
        parinfo2[6]['step']      = 0.0105
        parinfo2[5]['limits']    = (0.0, 1.0)
        parinfo2[6]['limits']    = (0.02, 0.2)
        parinfo2[8]['fixed']=True ##fix the cubic/quartic terms to zero in these two lines
        parinfo2[9]['fixed']=True
        parinfo2[8]['limits'] = (0.000001, windowsize-0.00000001)
        parinfo2[9]['limits'] = (-0.1, 0.1)

        ##if arc length is not going to be used in the fit, fix those parameters
        if use_arclength == False:
            parinfo2[3]['fixed'] = True
            parinfo2[4]['fixed'] = True
            parinfo2[7]['fixed'] = True

        if (use_raw == False) & (use_arclength == True): parinfo2[4]['fixed'] = True #make arclength fit linear when using corrected curves and arclength

        ##run on a grid of transit durations for the notch, for things that are planet like in depth
        lgrid = np.array([0.75, 1.0, 2.0, 4.0])/24.0
        numpars = len(parinfo2) - np.sum([parinfo2[dude]['fixed'] for dude in range(len(parinfo2))])

        if len(wdat) < numpars+1+5:##if hardly any points in window, just use the initial fit
            modpoly = np.polyval(initpoly, wdat.t)
            pos_outlier = np.array([], dtype=int)
            pars = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            #print( 'in nodata fit')
        else: ##if enough points in this window to do the full notch filter fit, go fit it
            #print 'in full fit'


            ##if switch to set, don't try the single point transit (45 mins)
            ##also adjust related fit limit
            if resolvable_trans == True:
                lgrid = np.array([1.0, 2.0, 4.0])/24.0
                parinfo2[6]['limits'] = (0.042, 0.2)
            ##fix the duration in the grid fitting
            parinfo2[6]['fixed'] = True
            bestpars             = pstart*0.0
            current_chi2         = np.inf ##not going to store anything except the current best solution on the grid
            for l in range(len(lgrid)):
                pstart[6] = lgrid[l]
                dfit2, extrares = mpyfit.fit(transit_window_slide_pyfit3, pstart, args=(wdat.t, wdat.fraw, error, wdat.s, thistime), parinfo=parinfo2, maxiter=200) ##args = t, fl, sig_fl, s, ttime
                if extrares['bestnorm'] < current_chi2:
                    bestpars     = dfit2*1.0
                    current_chi2 = extrares['bestnorm']*1.0
            ##Now that best grid solution is found, unfix the transit duration paramater
            parinfo2[6]['fixed'] =True

            ##run the fit with no fixed parameters but from the best grid point result from above
            pstart  = bestpars


            pars, extrares = mpyfit.fit(transit_window_slide_pyfit3, pstart, args=(wdat.t, wdat.fraw, error, wdat.s, thistime), parinfo=parinfo2, maxiter=200)
            themod, modpoly, modnotch, modspot, modresid = transit_window_slide_pyfit3(pars, args=(wdat.t, wdat.fraw, error, wdat.s, thistime), model=True)

            ##remove high outliers (like flares) and run again, do this based on a statistical measure like rms
            ##loop a few times to converge on an outlier-less solution
            ##run the outlier rejection 5 times, that should be enough....
            pos_outlier = np.zeros(0, int)
            cliperr     = error*1.0
            oldnum=0

            for lll in range(0, 5):
                koff     = np.where(np.isinf(cliperr) == False)[0]
                rms      = np.sqrt(np.mean((wdat.fraw[koff]-themod[koff])**2))
                rmsoff   = (wdat.fraw - themod)/rms
                knownbad = np.where(np.isinf(cliperr))[0]
                if np.isnan(rms) | (rms ==0): break


                ##identify outliers
                new_outl    = np.where((rmsoff > cliplimu) | (rmsoff < -cliplim))[0]
                pos_outlier = np.unique(np.append(pos_outlier, new_outl))
                cliperr[:]  = rms*1.0
                cliperr[pos_outlier] = np.inf   ##set outlier point uncertainties to effective infinity
                cliperr[knownbad]    = np.inf   ##set outlier point uncertainties to effective infinity

                if (len(pos_outlier) > oldnum) | (lll==0): ##dont refit unless theres a new point flagged as an outlier.
                    fa      = {'fl':wdat.fraw, 'sig_fl':cliperr, 't':wdat.t, 's':wdat.s, 'ttime':thistime}
                    args=(wdat.t, wdat.fraw, error, wdat.s, thistime)
                    pars, thisfit = mpyfit.fit(transit_window_slide_pyfit3, pstart, args=(wdat.t, wdat.fraw, cliperr, wdat.s, thistime), parinfo=parinfo2)
                    themod, modpoly, modnotch, modspot, modresid = transit_window_slide_pyfit3(pars, args=(wdat.t, wdat.fraw, cliperr, wdat.s, thistime), model=True)

                oldnum = len(pos_outlier)
            ##calculate the final model chi2 here, need to recalc because the rms number might have changed after the last outlier rejection and model fit
            modchi2 = np.sum(((themod-wdat.fraw)/cliperr)**2)

            ##try model with no transit, i.e fix the depth to zero
            parinfo2[5]['fixed'] = True
            parinfo2[8]['fixed'] = True
            parinfo2[9]['fixed'] = True
            nullstart           = pars.copy()
            nullstart[5]        = 0.0
            nullstart[8]        = windowsize/2
            nullstart[9]        = 0.0

            nullfit, nullextra = mpyfit.fit(transit_window_slide_pyfit3, nullstart, args=(wdat.t, wdat.fraw, cliperr, wdat.s, thistime), parinfo=parinfo2)
            nullmod, nullpoly, dummy, nullspot, nullresid = transit_window_slide_pyfit3(nullfit, args=(wdat.t, wdat.fraw, cliperr, wdat.s, thistime), model=True)
            nullchi2 = np.sum(nullresid**2)

            ##what are the two model bayesian information criteria?
            numpars = len(parinfo2) - np.sum([parinfo2[dude]['fixed'] for dude in range(len(parinfo2))])
            npoints  = len(np.where(cliperr < 0.99)[0])
            bicnull  = nullchi2 + (numpars-2)*np.log(npoints)
            bicfull  = modchi2  + numpars*np.log(npoints)
            #import pdb
            #pdb.set_trace()
            ##make figures or check things?
            #if i >= -1:
                ##plotting moviemaking stuff, usually not run.
                #import pdb
                #import matplotlib.pyplot as plt
                #pdb.set_trace()
                #plt.plot(wdat.t, wdat.fraw, '.')
                #plt.plot(wdat.t, themod, 'r')
                #plt.plot(wdat.t, modpoly, 'g')
                #plt.show()
                #import pdb
                #pdb.set_trace()
                # mmm = np.zeros(len(wdat.t), dtype=int)
    #             mmm[pos_outlier] = -1
    #             oks = np.where((mmm >= 0))[0]
    #             ax1 = plt.subplot(211)
    #             plt.plot(wdat.t[oks], wdat.fraw[oks], 'ko', label='K2 Data')
    #             plt.plot(wdat.t[ttt], wdat.fraw[ttt], 'm*', markersize=18)
    #             plt.plot(wdat.t, themod, 'r', label = 'Poly + Notch')
    #             plt.plot(wdat.t, modpoly, 'r-.')
    #             plt.plot(wdat.t, nullpoly, '--g', label='Poly')
    #             plt.plot(wdat.t[pos_outlier], wdat.fraw[pos_outlier], 'rx', markeredgecolor='r', mew=2)
    #             if i <= 460:  plt.legend(loc='upper right')
    #             ax1.set_ylabel('Relative Flux')
    #             ax1.ticklabel_format(useOffset=False, style='plain')
    #
    #             ax2 = plt.subplot(212)
    #             oks2 = np.where((fittimes < fittimes[i]))[0]
    #             plt.plot(fittimes[oks2]-fittimes[0], detrend[oks2], 'k.')
    #             plt.plot(fittimes[wind[ttt]]-fittimes[0], wdat.fraw[ttt]/modpoly[ttt], '*m', markersize=18)
    #             beyond = np.where(fittimes > fittimes[i])[0]
    #             plt.plot(fittimes[beyond]-fittimes[0], data.fcor[beyond], 'g.')
    #             ax2.set_ylabel('Relative Flux')
    #             ax2.set_xlabel('Time (days)')
    #             if i <= 460:
    #                 ax2.set_xlim([5, 16])
    #                 ax2.set_ylim([0.985, 1.02])
    #             if i >460:
    #                 ax2.set_ylim([0.975, 1.02])
    #                 ax2.set_xlim([0.00, fittimes[-1]-fittimes[0]])
    #             ax2.ticklabel_format(useOffset=False, style='plain')
    #             if i <= 460:
    #                 for nnn in range(100): plt.savefig('moviemake/'+str(i)+'_'+str(nnn)+'_movmake.png')
    #             if i > 460:
    #                 plt.savefig('moviemake/'+str(i)+'_0'+'_movmake.png')
    #            #import pdb
    #            #pdb.set_trace()
    #
    #            plt.clf()


            ##if the bayesian information criterion for the full and null models provide evidence against the transit notch, 
            ##just use the null model for the fit, the difference required to believe a transit notch is requires is by
            ##default 0, which is very inclusive, but can be tuned if you know what you're looking for.
            ##deltaBIC > 0 means notch is favoured.
            dbic[i] = bicnull - bicfull
            if dbic[i] < deltabic:
                modpoly = nullmod ##if notch not required, use the null model
                ##pars[5] = 0.0     ##set to zero for later storage, actually retain it for record keeping
            stoppoint = 1500
            if  i >= stoppoint: ##for Aaron to look at whats getting notched, set i>-1 to see everything

                print( fittimes[i])
                print( bicnull-bicfull)
                #print pars[5]
                #print np.min(cliperr)
                #print rms

                if i == stoppoint:
                    import pdb
                    import matplotlib.pyplot as plt
                    plt.ion()

                plt.plot(wdat.t, wdat.fraw, '.')
                plt.errorbar(wdat.t, wdat.fraw, yerr = cliplim*0.0+rms, fmt='b.')
                plt.plot(wdat.t, themod, 'g')
                plt.plot(wdat.t, modpoly*modspot, 'k')
                plt.plot(wdat.t, nullmod, 'r')
                plt.plot(wdat.t[ttt], wdat.fraw[ttt], 'm*', markersize=18)
                plt.plot(wdat.t[pos_outlier], wdat.fraw[pos_outlier], 'rx', markeredgecolor='r', mew=2)

                #plt.show()
                pdb.set_trace()
                plt.clf()
        ##save things we care about, like a detrended curve and a transit depth fit for this point in time
        ttt           = np.where(wdat.t == thistime)[0]
        detrend[i]    = wdat.fraw[ttt]/modpoly[ttt]
        polyshape[i]  = modpoly[ttt]*1.0
        depthstore[i] = pars[5]*1.0
        ##make sure outliers get flagged appropriately
        badflag[wind[pos_outlier]] = 1
        badflag[wind[flare]] = 2
        if cleanmask[0] != -1:
            intrans = np.where(cleanmask == 99)[0]
            badflag[intrans] = 0

    ##spit output
    return fittimes, [depthstore, dbic], detrend, polyshape, badflag


##the LOCoR pipeline
##for now its called rcomb because I dont feel like changing all the calls.
##
def rcomb(data, prot, rmsclip=3.0, aliasnum=2.0, cleanmask=[-1, -1]):
    '''
    The Locally Optimized Combination of Rotations detrending method

    Inputs:
    (1) data: The data recarray that this code uses for passing data around
    (2) prot: The rotation period of this star measured by you elsewhere

    Optional Inputs:
    (1) rmsclip: Sigma-clipping level for outliers in the fit iterations default is 3
    (2) aliasnum: Period to alias the rotation period (prot) to. Needed when each period doesnt contain
    sufficient data for a good fit. Default is 2 days which works robustly for K2 long cadence data. Set to
    a number less than prot to not alias at all, e.g . set to -1.0.
    (3) cleanmask: binary mask to remove a set of points from the fitting. Great for masking over a transit signal that you
    dont want influencing the fit. Default is [-1, -1] which turns it off

    Outputs:
    (1) Times: The input time axis from data
    (2) Dummy: This is a dummy output of zeros to match the notch filter output format for compatibility
    (3) detrend: detrended lightcurve.f
    (4) finalmodel: the model used to detrend the input lightcurve
    (5) badflag: integer flags for each datapoint in data with 0=masked as outlier in iterations
        1=fine, 2=strong positive outlier or other suspect point.

    '''

    ##print 'aliasnum ' + str(aliasnum) ##testing line
    ##!!initial organizing
    ##calculate phases for this rotation period
    phase  = calc_phase(data.t, prot, data.t[0])
    ##locate each period, label it
    plabel = np.zeros(len(data.t), int)
    point  = np.where(phase - np.roll(phase, 1) < 0.0)[0]
    for i in range(len(point)-1): plabel[point[i]:point[i+1]] = i+1
    pers = np.unique(plabel)
    thelc = data.fcor*1.0
    ##offset each period to better line up with others, basically median divide each period
    for i in range(len(pers)):
        bnd = np.where(plabel == pers[i])[0]
        thelc[bnd]  = thelc[bnd]/np.nanmedian(thelc[bnd])



    ##!!Outlier Rejection
    ##now remove any outliers using lcbin, pretty simple (just marking them really)
    lbin, binsizerm, siglbin, goodind, phasebin = lcbin(phase, thelc, 20, userobustmean=True)
    badflag = np.zeros(len(data.t), int)+0
    badflag[goodind] = 1
    ##any points less than 0, flag them as bad now (failure mode output of K2sff)
    qwe = np.where(data.fcor <= 0.0)[0]
    badflag[qwe] = 2

    ##for the case of masking a known transit out of the fit::
    if cleanmask[0] != -1:
        maskthese = np.where(cleanmask == 99)[0]
        badflag[maskthese] = 0


    finalmodel = thelc*0.0
    detrend    = thelc*0.0

    ##should we alias up the rotation period?
    ##If Prot < 2 (default number) alias it up to at least 2, also correct the phases to match new period
    newprot = prot*1.0
    if prot < aliasnum:
        pfactor    = int(np.ceil(aliasnum/prot))
        newprot    = prot*pfactor
    ##!Reorganize to new rotation period
    ##recalc phases and period labelling to newprot
    phase = calc_phase(data.t, newprot, data.t[0])
    plabel = np.zeros(len(data.t), int)
    point  = np.where(phase - np.roll(phase, 1) < 0.0)[0]
    for i in range(len(point)-1): plabel[point[i]:point[i+1]] = i+1

    ##stack each period of data and correct any bulk offsets again, 
    ##usually a very minor correction
    liblc = ()
    libph = ()
    libflag = ()
    keepindex = ()
    pers = np.unique(plabel)
    for i in range(len(pers)):
        bnd = np.where(plabel == pers[i])[0]
        keepindex += (bnd*1, )
        thisperiod = thelc[bnd]/np.nanmedian(thelc[bnd])
        liblc  += (thisperiod*1.0, )
        libph  += (phase[bnd]*1.0, )
        libflag +=(badflag[bnd]*1, )
    liblc     = np.array(liblc, dtype='object')
    libph     = np.array(libph, dtype='object')
    libflag   = np.array(libflag, dtype='object')
    keepindex = np.array(keepindex, dtype='object')
    stackmod  = liblc*0.0

    ##we now have a clean library of rotation periods with outliers flagged up.

    ##now do the actual fit, go period by period
    for i in range(len(liblc)):
        ##extract the period of interest, along with outlier flags
        thisphase, thislc, thisflag = libph[i]*1.0, liblc[i]*1.0, libflag[i]*1

        ##if lots of points are somehow called outliers (some failure mode of robust mean)
        ##just use them all. This happens rarely.
        ##Also note this never changes badflg=2 points, they stay bad forever
        pointsin = np.where(thisflag == 0)[0] ##number of bad points
        ##if more than half bad, call them all good
        if len(pointsin) > len(thisflag)/2: thisflag[pointsin] =1
        pointsin  = np.where(thisflag == 1)[0] ##how many we have now?
        reallybad = np.where(thisflag == 2)[0]
        thisflag[reallybad] = 0 ##reset all the failpoints to 0 flags
        ##ID the non-fitting periods
        oth = np.where(np.arange(len(liblc)) != i)[0]
        olc = ()




        ##Limit the number of periods to use so fit is not over specified.
        ##Ensures A matrix is non-sigular and invertible. This only matters for
        ##a partial period e.g. at the end of the dataset.
        ##Currently taking the nearest rotation periods within this limit.
        sizelim = len(pointsin)-3
        nearest = np.argsort(np.absolute(oth-i))
        oth = oth[nearest[0:sizelim]]

        ##now be careful, if the expected number of points is too small
        ##just return the input values, also set badflag to 2 so they are removed from
        ##BLS search later.
        ##this happens when the partial period at end of dataset is only a couple of points

        if (sizelim >= 1) & (-np.nanmin(thislc) + np.nanmax(thislc) > 0.00000000000001):




            ##now interpolate all the reference periods onto the phase-scale of the
            ##period we are fitting, being careful to not use the outliers.
            for j in range(len(oth)):
                keep = np.where(libflag[oth[j]] == 1)[0]
                if (len(keep) > 5) & (len(libflag[oth[j]]) > len(thisflag)*2./3.): ##only keep a period if it has more than half its points as fine
                    if (-np.nanmin(liblc[oth[j]][keep]) + np.nanmax(liblc[oth[j]][keep]) > 0.00000000000001): olc  += (np.interp(thisphase, libph[oth[j]][keep], liblc[oth[j]][keep]), )
            olc = np.array(olc)

            ##now check that the reference library actually has something in it.
            if len(olc) > 0:
                rrr=0
                while rrr < 3: ##do this at most three times, removing outliers each time.
                    ##reference library is ready to use now.
                    ##populate the A matrix, masking over outliers in sum with flag array
                    Amat = np.dot(olc*thisflag, olc.T)
                    ##populate b vector, again masking over outliers in the sum
                    bvect = np.dot(olc*thisflag, thislc)
                    ##do the inverse multiplication
                    coeffs = np.dot(bvect, np.linalg.inv(Amat))
                    themodel = 0.0
                    for j in range(len(olc)): themodel += olc[j]*coeffs[j] ##Sum up the model




                    #now find the rmsoffset
                    totest = np.where(thisflag==1)[0]
                    mad = np.median(np.absolute((thislc[totest]/themodel[totest]-1)))/0.67
                    #rms = np.sqrt(np.mean((thislc/themodel*thisflag - thisflag)**2))
                    offset = (thislc*thisflag/themodel-1)/mad*thisflag
                    outl = np.where(np.absolute(offset) > rmsclip)[0]



                    #print mad, rms
                    ##set the flag on this outlier to 0
                    if len(outl) > 0:
                        thisflag[outl] = 0 ##now repeat the process
                        badflag[keepindex[i][outl]] = 0 ##also store for long-term output
                    if len(outl) == 0 : rrr = 5
                    ##now check that we have enough points to actually do another fit
                    ## do this by truncating the library period array
                    numberin = len(np.where(thisflag == 1)[0])
                    if numberin <= len(olc):
                        rrr = 5
                        print( 'limit hit for outliers')
                    rrr +=1

                ##what happens to the model at point where thisflag==0? should really remove the model
                ##and replace with interpolation from of actually fit points
                nofit = np.where(thisflag == 0)[0]
                if len(nofit) > 0 :
                    donefit = np.where(thisflag == 1)[0]
                    themodel[nofit] = np.interp(thisphase[nofit], thisphase[donefit], themodel[donefit])


            else: ##if no points, (fail case for k2sff) flag everything as bad, model is null
                themodel  = thislc*0.0+1.0 ##don't alter the points because fit failed
                badflag[keepindex[i]] = 1  ##flag as bad points ##this causes other parts of the code to fail, better to just flag everything as 1 in this case

        else: ##case where there's not enough points in this phaseing, just call these points bad going foward
            themodel  = thislc*0.0+1.0 ##don't alter the points because fit failed
            badflag[keepindex[i]] = 1  ##flag as bad points ##this causes other parts of the code to fail, better to just flag everything as 1 in this case

        ##put this period's fit into the appropriate spot in full array
        stackmod[i] = themodel*1.0
        finalmodel[keepindex[i]] = themodel*1.0
        detrend[keepindex[i]] = thislc/themodel


    ##now find all outliers that are positive and flag them as bad
    badflag[np.where((badflag ==0) & (detrend > 1.0))[0]] = 2
    return data.t, data.t*0.0, detrend, finalmodel, badflag


##experimental, do use unless you want to waste time, or you are aaron
def locor2(data, prot, rmsclip=3.0, aliasnum=2.0, cleanmask=[-1, -1], deltabic=0.0):

    ##!!initial organizing
    ##calculate phases for this rotation period
    phase  = calc_phase(data.t, prot, data.t[0])
    ##locate each period, label it
    plabel = np.zeros(len(data.t), int)
    point  = np.where(phase - np.roll(phase, 1) < 0.0)[0]
    for i in range(len(point)-1): plabel[point[i]:point[i+1]] = i+1
    pers = np.unique(plabel)
    thelc = data.fcor*1.0
    ##offset each period to better line up with others, basically median divide each period
    for i in range(len(pers)):
        bnd = np.where(plabel == pers[i])[0]
        thelc[bnd]  = thelc[bnd]/np.nanmedian(thelc[bnd])



    ##!!Outlier Rejection
    ##now remove any outliers using lcbin, pretty simple (just marking them really)
    lbin, binsizerm, siglbin, goodind, phasebin = lcbin(phase, thelc, 20, userobustmean=True)
    badflag = np.zeros(len(data.t), int)+0
    badflag[goodind] = 1
    lbininterp = np.interp(phase, phasebin, lbin)
    tokeep = np.where((badflag == 0) & (lbininterp-thelc > 0.0))[0]
    badflag[tokeep] = 1
    #import pdb
    #import matplotlib.pyplot as plt
    #pdb.set_trace()
    ##any points less than 0, flag them as bad now (failure mode output of K2sff)
    qwe = np.where(data.fcor <= 0.0)[0]
    badflag[qwe] = 2

    ##for the case of masking a known transit out of the fit::
    if cleanmask[0] != -1:
        maskthese = np.where(cleanmask == 99)[0]
        badflag[maskthese] = 0


    finalmodel = thelc*0.0
    detrend    = thelc*0.0
    depthstore = thelc*0.0
    ##should we alias up the rotation period?
    ##If Prot < 2 alias it up to at least 2, also correct the phases to match new period
    newprot = prot*1.0
    if prot < aliasnum:
        pfactor    = int(np.ceil(aliasnum/prot))
        newprot    = prot*pfactor

    ##!Reorganize to new rotation period
    ##recalc phases and period labelling to newprot
    phase = calc_phase(data.t, newprot, data.t[0])
    plabel = np.zeros(len(data.t), int)
    point  = np.where(phase - np.roll(phase, 1) < 0.0)[0]
    for i in range(len(point)-1): plabel[point[i]:point[i+1]] = i+1

    ##stack each period of data and correct any bulk offsets again, 
    ##usually a very minor correction
    liblc = ()
    libph = ()
    libflag = ()
    keepindex = ()
    pers = np.unique(plabel)
    for i in range(len(pers)):
        bnd = np.where(plabel == pers[i])[0]
        keepindex += (bnd*1, )
        thisperiod = thelc[bnd]/np.nanmedian(thelc[bnd])
        liblc  += (thisperiod*1.0, )
        libph  += (phase[bnd]*1.0, )
        libflag +=(badflag[bnd]*1, )
    liblc     = np.array(liblc)
    libph     = np.array(libph)
    libflag   = np.array(libflag)
    keepindex = np.array(keepindex)
    stackmod  = liblc*0.0

    ##we now have a clean library of rotation periods with outliers flagged up.

    ##now do the actual fit, go period by period
    for i in range(len(liblc)):
        print( i)
        ##extract the period of interest, along with outlier flags
        thisphase, thislc, thisflag = libph[i]*1.0, liblc[i]*1.0, libflag[i]*1

        ##if lots of points are somehow called outliers (some failure mode of robust mean)
        ##just use them all. This happens rarely.
        ##Also note this never changes badflg=2 points, they stay bad forever
        pointsin = np.where(thisflag == 0)[0] ##number of bad points
        ##if more than half bad, call them all good
        if len(pointsin) > len(thisflag)/2: thisflag[pointsin] =1
        pointsin  = np.where(thisflag == 1)[0] ##how many we have now?
        reallybad = np.where(thisflag == 2)[0]
        thisflag[reallybad] = 0 ##reset all the failpoints to 0 flags
        ##ID the non-fitting periods
        oth = np.where(np.arange(len(liblc)) != i)[0]
        olc = ()




        ##Limit the number of periods to use so fit is not over specified.
        ##Ensures A matrix is non-sigular and invertible. This only matters for
        ##a partial period e.g. at the end of the dataset.
        ##Currently taking the nearest rotation periods within this limit.
        sizelim = len(pointsin)-3
        nearest = np.argsort(np.absolute(oth-i))
        oth = oth[nearest[0:sizelim]]

        ##now be careful, if the expected number of points is too small
        ##just return the input values, also set badflag to 2 so they are removed from
        ##BLS search later.
        ##this happens when the partial period at end of dataset is only a couple of points
        periodmodel = thislc*0.0+1.0
        perioddepth = thislc*0.0
        if (sizelim >= 1) & (-np.nanmin(thislc) + np.nanmax(thislc) > 0.00000000000001):




            ##now interpolate all the reference periods onto the phase-scale of the
            ##period we are fitting, being careful to not use the outliers.
            for j in range(len(oth)):
                keep = np.where(libflag[oth[j]] == 1)[0]
                if (len(keep) > 5) & (len(libflag[oth[j]]) > len(thisflag)*2./3.): ##only keep a period if it has more than half its points as fine
                    if (-np.nanmin(liblc[oth[j]][keep]) + np.nanmax(liblc[oth[j]][keep]) > 0.00000000000001): olc  += (np.interp(thisphase, libph[oth[j]][keep], liblc[oth[j]][keep]), )


            olc += (olc[0]*0.0, ) ##add the final part for the hat function to be plonked into
            origolc = np.array(olc)  ##make it an easier to access numpy array
            ##now check that the reference library actually has something in it.
            if len(olc) > 1:
                phasewidths = np.array([0.75, 1.0, 2.0, 4.0, 0.0])/24.0/newprot

                for cad in range(len(thisphase)):##go point by point in this rotation pseudo-period
                    #print cad
                    pointphase = thisphase[cad]


                    ##first try fitting with the different window sizes
                    buseflag = thisflag*0 -1
                    thischi2 = np.inf
                    bestcoeffs = 0.0*len(origolc) - 1
                    bestwidth  = -100.0
                    for wwind, www in enumerate(phasewidths):
                        ##add hat shape transit function to OLC
                        ##make sure everythings modded right to capture full transit
                        useflag = thisflag*1
                        olc = origolc*1.0
                        olc[-1, :] = 0.0
                        if wwind < len(phasewidths)-1:
                            shifter = 0.5-pointphase
                            wpoints = np.where((np.mod(thisphase+shifter, 1.0) < 0.5+www/2.0) & (np.mod(thisphase+shifter, 1.0)> 0.5-www/2.0))[0]
                        else: wpoints = np.array([], dtype=int)
                        olc[-1, wpoints] = 1.0
                        if len(wpoints) == 0: olc = olc[:-1] ##if null model remove notch part of matrix olc
                        if np.sum(olc[-1]*useflag) == 0 : olc = olc[:-1] ##if all points in notch are called outliers, remove notch part of olc
                        #if cad == 9:
                        #    import pdb
                        #    pdb.set_trace()
                        ##make up new flag arrays



                        rrr=0
                        while rrr < 3: ##do this at most three times, removing outliers each time.
                            ##reference library is ready to use now.
                            ##populate the A matrix, masking over outliers in sum with flag array
                            Amat = np.dot(olc*useflag, olc.T)
                            ##populate b vector, again masking over outliers in the sum
                            bvect = np.dot(olc*useflag, thislc)
                            ##do the inverse multiplication
                            coeffs = np.dot(bvect, np.linalg.inv(Amat))
                            themodel = 0.0
                            for j in range(len(olc))  :  themodel += olc[j]*coeffs[j] ##Sum up the model
                            polymod = 0.0
                            for j in range(len(olc)-1)  : polymod += olc[j]*coeffs[j] ##Sum up the model

                            #now find the rmsoffset
                            totest  = np.where(useflag==1)[0]
                            mad     = np.median(np.absolute((thislc[totest]/themodel[totest]-1)))/0.67
                            offset  = (thislc*thisflag/themodel-1)/mad*thisflag
                            outl    = np.where(np.absolute(offset) > rmsclip)[0]
                            modchi2 = np.sum((themodel[totest]-thislc[totest])**2/mad**2)


                            #print mad, rms
                            ##set the flag on this outlier to 0
                            if len(outl) > 0:
                                useflag[outl] = 0 ##now repeat the process
                                if np.sum(olc[-1]*useflag) == 0 : olc = olc[:-1] ##if all points in notch are called outliers, remove notch part of olc
                                 ##also store for long-term output
                            if len(outl) == 0 : rrr = 5
                            ##now check that we have enough points to actually do another fit
                            ## do this by truncating the library period array
                            numberin = len(np.where(useflag == 1)[0])
                            if numberin <= len(olc):
                                rrr = 5
                                print( 'limit hit for outliers')
                            rrr +=1
                        #print modchi2
                        if (modchi2 < thischi2) & (wwind < len(phasewidths)-1): ##if this model is better than previous models, save it's information
                            buseflag   = useflag*1
                            thischi2   = modchi2*1.0
                            bestcoeffs = coeffs*1.0
                            bestwidth  = www*1.0
                            bestmodel = themodel*1.0
                            bestpoly  = polymod*1.0
                        if wwind == len(phasewidths)-1:
                            nullflag   = useflag*1
                            nullchi2   = modchi2*1.0
                            nullcoeffs = coeffs*1.0
                            nullwidth  = 0.0
                            nullmodel  = themodel*1.0

                    bicnull = nullchi2 + (len(origolc)-1)*np.log(np.sum(nullflag))
                    bicfull = thischi2 + len(origolc)*np.log(np.sum(buseflag))
                    if (bicnull-bicfull > deltabic) & (bestcoeffs[-1] < 0.0):
                        usepoly = bestpoly*1.0
                        usedepth = -1.0*bestcoeffs[-1]
                        useflag = buseflag*1
                    else:
                        usepoly = nullmodel*1.0
                        usedepth = 0.0
                        useflag = nullflag*1
                    periodmodel[cad] = usepoly[cad]*1.0
                    perioddepth[cad] = usedepth*1.0
                    #import pdb
                    #import matplotlib.pyplot as plt
                    #pdb.set_trace()

                    import pdb
                    import matplotlib.pyplot as plt
                    qwe = np.where(perioddepth > 0.0)[0]
                    plt.plot(thisphase, thislc, 'b.')
                    plt.plot(thisphase, bestpoly, 'g.')
                    plt.plot(thisphase, bestmodel, 'r.')
                    qwe = np.where(buseflag == 0)[0]
                    plt.plot(thisphase[qwe], thislc[qwe], 'xm')

                    plt.plot(thisphase, nullmodel, 'm.')
                    plt.plot(thisphase[cad], thislc[cad], 'ob')
                    plt.show()
                    pdb.set_trace()

            else: ##if no points, (fail case for k2sff) flag everything as bad, model is null
                periodmodel  = thislc*0.0+1.0 ##don't alter the points because fit failed
                badflag[keepindex[i]] = 2  ##flag as bad points

        else: ##case where there's not enough points in this phaseing, just call these points bad going foward
            periodmodel  = thislc*0.0+1.0 ##don't alter the points because fit failed
            badflag[keepindex[i]] = 2  ##flag as bad points

        ##put this period's fit into the appropriate spot in full array
        stackmod[i] = periodmodel*1.0
        finalmodel[keepindex[i]] = periodmodel*1.0
        detrend[keepindex[i]] = thislc/periodmodel
        depthstore[keepindex[i]] = perioddepth*1.0
        import pdb
        import matplotlib.pyplot as plt
        qwe = np.where(perioddepth > 0.0)[0]
        plt.plot(thisphase, thislc, 'b.')
        plt.plot(thisphase[qwe], thislc[qwe], 'g.')
        plt.plot(thisphase, periodmodel, 'r.')
        plt.show()


        #pdb.set_trace()

    ##now find all outliers that are positive and flag them as bad
    badflag[np.where((badflag ==0) & (detrend > 1.0))[0]] = 2
    return data.t, depthstore, detrend, finalmodel, badflag


##SNR calculation for BLS power spectra that uses a median-absolut-deviation estimator for the
##half normal distribution stdev, works fairly well for most periods. There could be issues if you search a huge span of periods where red noise effects statistics.
def bls_power_analysis_snr(pgrid, power):
    ##bins per period is important here: about 1 bin for every 2 days in period seems fine once logged. this is what I did for K2
    powbin, binsizerm, pebinm, goodind, perbin = lcbin(np.log(pgrid), power, int(np.floor(np.max(pgrid)/2)), userobustmean=True)
    trendpow = np.interp(np.log(pgrid), perbin, powbin)
    flatpow  = power - trendpow
    snr      = flatpow/(np.median(np.absolute(np.median(flatpow) - flatpow))/0.6745)
    bestspot = np.argmax(snr)
    bestp    = pgrid[bestspot]
    bestsnr  = snr[bestspot]
    return bestp, bestsnr



##run to search for transit like signals in a detrended lightcurve, basically a stand-alone copy of what detrend_k2 does
##data has to look like the recarrays used everywhere else in this code
def bls_transit_search(data, detrend, badflag, rmsclip=3.0, snrcut=7.0, cliplow
                       = 30.0, binn=300, period_matching=[-1, -1, -1],
                       searchmax=30.0, searchmin=1.0000001, mindcyc=0.005,
                       maxdcyc=0.3, freqmode='standard', datamode='standard'):
    ##assess the output flags from the detrending
    mmm = np.ones(len(data.t), dtype=int)
    bad = np.where(((data.s<0.0) | (data.s>8)) & (detrend<0.99) | (detrend <=0.0))[0]
    mmm[bad] = 0

    ##remove other high points, they can hide transits from BLS
    extracut1 = np.percentile(detrend, 99.9)##don't plot the highest 0.1% of points, they are either garbage or not important for us now
    extracut2 = np.percentile(detrend, 0.1)##don't plot the lowest 0.1% of points, they are either garbage or not important for us now
    if datamode == 'bic' : extracut2 = 0.0

    qwe = np.where((detrend < extracut1) & (detrend > extracut2))[0]
    lcrms = np.sqrt(np.nanmean((1.0-detrend[qwe])**2))
    good = np.where((badflag < 2)  & (mmm == 1) & (detrend < rmsclip*lcrms+1.0))[0] ##(detrend > 1.0 - cliplow*lcrms)

    flar = np.where(badflag == 2)[0]
    pout = np.where(badflag == 1)[0]
    bad  = np.where(badflag == 3)[0]

    ##The transit search on the detrended lightcurve, using BLS.

    ##setup the bls stuff
    uvect = data.t.copy()*0.0
    vvect = data.t.copy()*0.0
    fstep = 0.00002
    if freqmode  == 'fine': fstep = 0.000002  ##for multi-campaign searches
    #if searchmax <= 1.0: fstep = 0.01 ##for USP mode, try not to over period, not sure this is needed anymore
    freq  = np.arange(1./searchmax, 1./searchmin, fstep)
    pgrid = 1.0/freq

    firstpower = uvect*1.0*0.0 ##set a firstpower for output even when the torun cut fails
    firstphase = uvect*1.0*0.0 ##set a firstpower for output even when the torun cut fails
    ##run the transit search in a loop, masking each detected signal
    ##10 should be more than enough
    mask   = np.ones(len(good), dtype=int)
    dp     = np.zeros(10)
    best_p = np.zeros(10)
    detsig = np.zeros(10)
    t0     = np.zeros(10)
    dcyc   = np.zeros(10)


    for cnt in range(10):
        torun = np.where(mask == 1)[0] ##region not masked out

        uvect[:] = 0.0
        vvect[:]=0.0


        if (len(torun) <= binn*2): ##check there's enough points to do a transit search
            dp     = dp[0:cnt+1]
            best_p = best_p[0:cnt+1]
            t0     = t0[0:cnt+1]
            detsig = detsig[0:cnt+1]
            dcyc = dcyc[0:cnt+1]
            break ## if we've mask so many points there's not enough for a binning, end.

        if (np.nanmax(data.t[good[torun]]) - np.nanmin(data.t[good[torun]]) < np.max(pgrid)+1.0): #check theres a decent span of time, separate to above due to error conditions
            dp     = dp[0:cnt+1]
            best_p = best_p[0:cnt+1]
            t0     = t0[0:cnt+1]
            detsig = detsig[0:cnt+1]
            dcyc = dcyc[0:cnt+1]
            break ## if lots of bad points, kill

        import bls

        #THIS is the BLS search now
        power, thisbest_p, best_pow, thisdp, qnum, in1, in2 = (
            bls.eebls(data.t[good[torun]], detrend[good[torun]],
                      uvect[good[torun]], vvect[good[torun]], len(freq),
                      freq[0], fstep, binn, mindcyc, maxdcyc)
        )
        if cnt == 0: firstpower = power*1.0 ##save the most raw power spectrum for outputting

        ##now is where we have to clean pgrid of found things
        pgrid=1/freq
        for ddd in range(cnt):
            #print ddd
            for aaa in range(1): ##also kill aliases
                matchgrid = np.where((pgrid < best_p[ddd]/(aaa+1)-2*timeres) | (pgrid > best_p[ddd]/(aaa+1)+2*timeres))[0]
                pgrid = pgrid[matchgrid]
                power = power[matchgrid]

        ##Determine the highest SNR peak by flattening first
        bbp, snr  = bls_power_analysis_snr(pgrid, power)
        best_p[cnt] = bbp*1.0
        detsig[cnt] = snr*1.0

        ##center the transit at phase 0.5 for ease of use later
        ##need to rerun the BLS on the best point from the good analysis to do this
        dpower, dthisbest_p, dbest_pow, dthisdp, dqnum, din1, din2 = (
            bls.eebls(data.t[good[torun]], detrend[good[torun]],
                      uvect[good[torun]], vvect[good[torun]], 1,
                      1.0/best_p[cnt], fstep, binn, mindcyc, maxdcyc)
        )
        dcyc[cnt] = dqnum*1.0
        dp[cnt]   = dthisdp*1.0 ##store the depth
        if din1 > din2: din1 -= binn

        rezero    = ((din1/2.0+din2/2.0)/binn-0.5)*best_p[cnt]
        t0[cnt]   = data.t[good[torun]][0]+rezero
        phase     = calc_phase(data.t, best_p[cnt], t0[cnt])
        if cnt == 0 : firstphase = phase*1.0 #if this is the first detection, store the phase for output for ease of use in other places
        ##exit the search loop if the current planet is below the SNR limit
        if ((snr <= snrcut) & (cnt >-1)) | (best_p[cnt] < np.min(pgrid)):
            dp     = dp[0:cnt+1]
            best_p = best_p[0:cnt+1]
            t0 = t0[0:cnt+1]
            detsig = detsig[0:cnt+1]
            dcyc = dcyc[0:cnt+1]
            break
        else: ##a detection, start with period matching for injection recovery if asked to, then mask the current signal and move on
            if period_matching[0] > 0.0:
                pdet = False
                ddet = False
                tdet = False
                ##Check the period is within 1% of input
                if (best_p[cnt]/period_matching[0] < 1.01) & (best_p[cnt]/period_matching[0] >0.99): pdet = True ##period match condition
                ##Check the depth is loosely ok (and not positive)
                if (dp[cnt]/period_matching[1] < 1.5)  & (dp[cnt]/period_matching[1] > 0.2): ddet = True ##depth match condition, super loose really
                if datamode == 'bic': ddet = True ##depth means nothing really when in deltabic space
                ##now check t0 is within 1% of the period, requires some moduloing:
                if np.mod(np.absolute(t0[cnt] + 0.5*best_p[cnt]-period_matching[2]), period_matching[0])/period_matching[0] <  0.01: tdet = True

                if (pdet == True) & (ddet == True) & (tdet == True): return 1, best_p[cnt], t0[cnt]+0.5*best_p[cnt], dp[cnt] ##we found the injected planet, no need to continue

            ##mask the detection and research, unless period_matching is happening
            ##in transit points are based on BLS qnum, with 50% added to ingress/egress to match binning issues
            trp = np.where((phase[good] < 0.5+dqnum*2.0) & (phase[good] > 0.5-dqnum*2.0))[0]
            #import pdb
           # pdb.set_trace()
           # trp=[] #just testing
            timeres = scipy.stats.mode(data.t-np.roll(data.t, 1))[0][0]

            mask[trp] = 0


    ##the case where we've hit the sensitivity limit and the injected planet has not been found yet.
    if period_matching[0] > 0.0: return 0, -1, -1, -1

    ##otherwise return everything else:

    return best_p, dp, t0, detsig, firstpower, 1.0/freq, firstphase, dcyc


def make_transitstamp_interactive(starname, time, detrend, bflag, period, t0, outdir='transitstamps/'):
    from gaussfit import gaussfit_mp
    from plotpoint import plotpoint
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    ##set matplotlibglobal params
    mpl.rcParams['lines.linewidth']   = 1.5
    mpl.rcParams['axes.linewidth']    = 2
    mpl.rcParams['xtick.major.width'] =2
    mpl.rcParams['ytick.major.width'] =2
    mpl.rcParams['ytick.labelsize'] = 15
    mpl.rcParams['xtick.labelsize'] = 15
    mpl.rcParams['axes.labelsize'] = 18
    mpl.rcParams['legend.numpoints'] = 1
    mpl.rcParams['axes.labelweight']='semibold'
    mpl.rcParams['font.weight'] = 'semibold'

    phase = calc_phase(time, period, t0)
    phasetime = (phase-0.5)*period
    extracut = np.percentile(detrend, 99.9)##don't plot the highest 0.1% of points, they are either garbage or not important for us now
    extracut2 = np.percentile(detrend, 100.-99.9)
    lcrms = np.sqrt(np.nanmean((1.0-detrend[np.where((detrend < extracut) & (detrend > extracut2))[0]])**2))
    good = np.where((bflag < 2) & (detrend >0.1) & (detrend < 1.5*lcrms + 1) & (detrend < extracut))[0]



    ##now plot things up for the user to interact with using the plotpoint class
    print( 'Click start of Ingress and end of egress')
    lpoint = plotpoint(phase[good], detrend[good])
    lpoint.getcoord()
    ingress = (lpoint.clickpoints[0][0] - 0.5)*period
    egress  = (lpoint.clickpoints[1][0] - 0.5)*period


    xrange = np.max([np.absolute(ingress), np.absolute(egress)])
    inx = np.where((phasetime[good] < xrange) & (phasetime[good] > -xrange))[0]


    startdep = 1.0 - np.min(detrend[good[inx]])
    gfit0 =np.array([startdep, 0.0001, xrange, 1.0, 0.0, 0.0])
    gfitpars, gfitcov, gfit = gaussfit_mp(phasetime[good], detrend[good], detrend[good]*0.01, gfit0, nocurve=True)
    fitdepth = gfitpars[0]


    ##now readjust the yrange and in-transit region again
    inx = np.where((phasetime[good] < xrange*2.0) & (phasetime[good] > -xrange*2.))[0]
    if len(inx) == 0:pdb.set_trace()
    binlc, binsize, ebin, goodind, binphase = k2sff.lcbin(phasetime[good[inx]], detrend[good[inx]], np.min([len(inx)/5, 40]), userobustmean=True)
    #pdb.set_trace()
    stdep = 1-np.min(detrend[good[inx[goodind]]])
    yrange = [1.0 - stdep*1.1, 1.5*lcrms+1]  ##for now this is probably fine
    if xrange*2 >= period/2.0: yrange = [np.min(detrend), yrange[1]]
    if starname == '211093684': yrange = [np.min(detrend), np.max(detrend)]

    #import pdb
    #pdb.set_trace()

    perstring = str(np.round(period, decimals=3))[0:5] + ' days'
    perstring2 = str(np.round(period, decimals=3))[0:5] + 'days'

    fig, ax = plt.subplots(1)
    ax.plot(phasetime[good]*24., detrend[good], 'ok', markersize=5)
    if np.max(phasetime[good[inx]]) - np.min(phasetime[good[inx]]) < period/2: ax.plot(binphase*24., binlc, 'ro', markeredgecolor='r', markersize=10) ##only plot the binned LC for planets not EB's
    ax.set_xlim([np.max([-xrange*2, np.min(phasetime[good])])*24, np.min([xrange*2, np.max(phasetime[good])])*24])
    ax.set_ylim(yrange)
    ax.set_ylabel('Relative Brightness')
    ax.set_xlabel('Phased Time (Hours)')
    ax.text(0.94, 0.05, 'EPIC '+starname + ' ' + 'P='+perstring, horizontalalignment='right', verticalalignment='top', fontweight='bold', transform = ax.transAxes, backgroundcolor='w')
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    plt.tight_layout()
    plt.savefig(outdir + 'EPIC'+starname+'_' + perstring2 +'_transitstamp.pdf')


    plt.show()
    plt.cla()
    plt.clf()

    plt.close("all")
    donefig = 1


def lomb_the_scargle(time, flux, prange=[0.1, 40.0], snrcut = 5.0, fullreturn=False):
    '''
    function to do a lomb-scargle periodogram and figure out the rotation period
    from a light curve
    '''
    from astropy.stats import LombScargle
    import astropy.units as u
    import scipy.signal
    pssize = 10000
    fbin, binsize, ebin, allgoodind, tbin = k2sff.lcbin(time, flux, 60, usemean=False, userobustmean=True, linfit=False)

    lspgram  = LombScargle(time[allgoodind], flux[allgoodind])
    freqs    = np.linspace(1.0/prange[1], 1.0/prange[0], pssize)

    lspower = lspgram.power(freqs, normalization='model')

    stepsize = (1/prange[0]-1/prange[1])/pssize*2
    ks = np.round(1.0/stepsize).astype(int)
    if np.mod(ks, 2) == 0: ks +=1
    mfiltpower = scipy.signal.medfilt(lspower, kernel_size=ks)

    cleansig = lspower-mfiltpower ##dividing can create peaks in weird cases

    rms = np.sqrt(np.mean((np.median(cleansig)-cleansig)**2))
    mad = np.median(np.absolute(np.median(cleansig)-cleansig))
    snr = cleansig/rms
    ##now find peaks
    peakind = np.argmax(snr)
    peaksnr = snr[peakind]

    period  = 1/freqs[peakind]
    if peaksnr < snrcut: period = 1000
    print( 'Period is ' + str(period) + ' days')

    if fullreturn == True: return cleansig, freqs, period, peakind, snr
    return period
