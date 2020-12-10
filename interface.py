import core
import lcfunctions
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

class target:
    def __init__(self, label):
        self.id = str(label)
        self.source = ''
        
    def load_data_tess(self,datafile,removebad = True,sector_label=''):
        '''
            This function opens a standard TESS LC downloaded from mast or elsewhere        
        '''
        hdu = fits.open(datafile)
        self.rawdata = hdu[1].data.copy()
        hdu.close()
        self.sourcetype = 'TESS'
        self.sourcefile = str(datafile)
        ##process to a more useful format:
        self.data = lcfunctions.LCconvertCTL(self.rawdata,removebad=removebad)
        self.sector=sector_label
        
    def load_data(time,flux,rawflux=[0],qual=[0],al=[0],source='user',sourcefile = 'user',sector_label='user'):
        '''
            Load data from user input, maybe their own thing?
        '''
        dl = len(lcdata)
        outdata         = np.recarray((dl,),dtype=[('t',float),('fraw',float),('fcor',float),('s',float),('qual',int),('detrend',float),('divisions',float)])
        outdata.t = time
        outdata.fcor=flux 
        outdata.fraw[:]    = 0 
        outdata.al[:]      = 0
        outdata.detrend[:] = 0
        outdata.qual[:]    = 0
        if len(rawflux) == len(flux): outdata.fraw = rawflux
        if len(al)      == len(flux): outdata.s    = al
        if len(qual)    == len(flux): outdata.qual = qual
        okok = np.where(np.isnan(outdata.fcor)==False)[0]
        outdata.fraw /= np.nanmedian(outdata.fraw)
        outdata.fcor /= np.nanmedian(outdata.fcor)
        outdata   = outdata[okok]
        keep      = np.where(outdata.qual == 0)[0]
        outdata.s = 0.0
        outdata   = outdata[keep]   
        self.data = outdata.copy()
        outdata   = 0.0
        self.rawdata = 0
        self.sourcetype = str(source)
        self.sourcefile = str(sourcefile)
        self.sector=sector_label

        
    def load_data_k2(datafile):
        print('not yet mate')
        
        
    def plotlc(self,figsize=(10,5),returnfigs=False,alpha=0.6,justdata=False):
        fig,ax = plt.subplots(figsize=figsize)
        ax.plot(self.data.t,self.data.fcor,'.',label='Corrected',zorder=100,alpha=alpha)
        ax.plot(self.data.t,self.data.fraw,'.',alpha=alpha)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Relative Brightness')
        
        if justdata==False:
            try:
                ax.plot(self.notch.t,self.notch.detrend,'.k',label='Notch detrended')
            except:
     #           print('No notch detrend to plot')
                dummy=0
            try:
                ax.plot(self.locor.t,self.locor.detrend,'.r',label='LOCoR detrended')
            except:
    #            print('No locor detrend')
                dummy=0

        ax.legend()

        if returnfigs == True: return fig,ax
        
    ##then we need attributes like run_detrend or something.
    
    def run_notch(self,window=0.5,mindbic=-1.0):
        
        notch = lcfunctions.run_notch(self.data,window=window,mindbic=mindbic)
        self.notch = notch.copy()
        self.notch_windowsize = window*1.0
        self.notch_mindbic = mindbic*1.0
        notch=0
        
        
    def run_locor(self,prot=None,alias_num=0.1):
        
        if not prot: 
            if hasattr(self,'prot') == False: 
                print("You haven't provided a rotation period mate!")
                print("either call target.run_locor(prot=X) in days, or do target.prot = X (in days) first then call target.run_locor()")
                print("Also, remember to provide a period to alias to, this is important for long cadence [Use alias_num=X days argument], for short cadence TESS maybe 0.1 is fine")
                return
            else: prot = self.prot
        
        locor = lcfunctions.run_locor(self.data,prot,alias_num=alias_num)
        
        self.locor = locor.copy()
        self.locor_prot = prot*1.0
        self.locor_alias = alias_num*1.0
        locor=0
        
        
    









