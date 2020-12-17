import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle,os,sys,glob,pdb
mpl.rcParams['lines.linewidth']   = 2
mpl.rcParams['axes.linewidth']    = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.labelsize']   = 10
mpl.rcParams['xtick.labelsize']   = 10
mpl.rcParams['axes.labelsize']    = 14
mpl.rcParams['legend.numpoints']  = 1
mpl.rcParams['axes.labelweight']  = 'semibold'
mpl.rcParams['axes.titlesize']    = 9
mpl.rcParams['axes.titleweight']  = 'semibold'
mpl.rcParams['font.weight']       = 'semibold'

import core


'''
    This file should contain functions for doing vetting checks on detections
'''


### Here's an easy first example
def rotation_alias_check(periods,prot,folds=10,margin=1):
    '''
        Takes a list of detected planet periods from the BLS search, and a rotation period of the star, and checks if the 
        detected planet periods alias to the rotation period.
        
        INPUTS:
        -------
        periods (floats): list of periods to check for aliasing to rotation, can be numpy array
        prot (float): The rotation period of the star to check against
        folds: integer number of rotation aliases to check against default is up to 10 times the rotation period
        margin: The fractional difference between from the aliased rotation period for a period in periods to 
                be considered consistent in percentages (default is +-1%).
                
        OUTPUTS:
        ---------
        rotation_flag (int): Array same size as input periods array, with zero if no match to a 
                            rotation alias, or the multiple of the rotation period if id does match.

    '''
    rotation_flag = np.zeros(len(periods), dtype=int)
    for i in range(folds+1):
        qwe = np.where(((prot*(i+1)/periods) > 1.0-margin/100.0) & (prot*(i+1)/periods < 1.0+margin/100.0))[0] ##+- 1 percent
        rotation_flag[qwe] = i+1
    return rotation_flag
    
def eoplot(wratpper
    
def eo_plot(data,detrend,badflag,period,t0,dcyc,id=999,outdir=''):
    '''
        Purpose: make even-odd plots for eyeball vetting.
    '''
    
    ##the code below that identifies even/odd regions expect transits happen at phase 0.5. so code that in
    t0 = t0-period/2.0
    deltat = np.where(-data.t+np.roll(data.t,-1) > period)[0]
    ttt = data.t.copy()
    for i in range(len(deltat)):            
        bridge = np.linspace(ttt[deltat[i]],ttt[deltat[i]+1],np.max(-ttt[deltat[i]]+ttt[deltat[i]+1])/period*4)
        bridge[0]+= 0.001
        bridge[-1]+= -0.001
        ttt =np.concatenate((ttt[0:deltat[i]] ,bridge,ttt[deltat[i]+1:]))
        detrend = np.concatenate((detrend[0:deltat[i]] ,bridge*0.0*np.nan,detrend[deltat[i]+1:]))
        badflag = np.concatenate((badflag[0:deltat[i]] ,bridge*0.0*np.nan,badflag[deltat[i]+1:]))
        deltat += len(bridge)-1

    phase = core.calc_phase(ttt,period,t0) ##This phases transits to phase=0, add 0.5 to phase to place it a 0.5


    ##flag even and odd points.
    deltaper = phase - np.roll(phase,-1)
    spots    = np.insert(np.where(deltaper > 0)[0],0,-1)
    start_time = data.t[spots[1]]
    eoflag = np.zeros(len(ttt),dtype=int)-1
    
     
    eoflag[0:spots[1]] = 0
    current = 0
    current_time = start_time*1.0
    counter = 0
    while current_time <= np.max(data.t):
        current +=1 
        current = current % 2
        current_time = start_time + counter*period
        rng = np.where((data.t>= current_time) & (data.t< current_time+period))[0]
        eoflag[rng] =current
        counter += 1
    
    ##eoflag now has flags for even or odd (0/1) for each point in the dataset. 
    ##flag corresponding transits
    even_trans = np.where((eoflag == 1) & (phase < 0.5+dcyc) & (phase > 0.5-dcyc))[0]
    odd_trans  = np.where((eoflag == 0) & (phase < 0.5+dcyc) & (phase > 0.5-dcyc))[0]
        
    all_trans = np.concatenate((even_trans,odd_trans))

    tight_ev  = np.where((eoflag == 1) & (phase < 0.505) & (phase > 0.495))[0]
    tight_odd = np.where((eoflag == 0) & (phase < 0.505) & (phase > 0.495))[0]

    

    ev  = np.where(eoflag == 1)[0]
    odd = np.where(eoflag == 0)[0]

    ##adjust phases so that the planet transits at zero, and phase spans [-0.5,0.5]
    eside = np.where(phase < 0.5)[0]
    iside = np.where(phase >= 0.5)[0]
    phase -= 0.5

    fig,ax = plt.subplots(1)
    ax.set_xlabel('Phase P=' + str(period)[0:5] + ' days')
    ax.set_ylabel('Relative Brightness')
    
   
    ax.plot(phase[ev],detrend[ev],'.C0', alpha=0.5,markersize=4,label='Even',zorder=1)
    ax.plot(phase[odd],detrend[odd],'.r', alpha=0.5,markersize=4,label='Odd',zorder=1)
    ax.legend()
    ax.set_xlim([-0.05,0.05])
    fig.tight_layout()
    
    if outdir != '':
        if outdir[-1] != '/': outdir += '/'
        if os.path.exists(outdir) == True:
            figname = outdir + 'target' + str(id)+'_Per_'+str(period)+'_T0_'+str(t0) + '_evenodd.pdf'
            fig.savefig(figname)
        else: 
            print("Output directory doesn't exist, unable to save figure!")
    return fig,ax
    
    
    
    
    
    