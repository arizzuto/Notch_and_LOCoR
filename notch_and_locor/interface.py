from notch_and_locor import core
from notch_and_locor import lcfunctions
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.recfunctions import stack_arrays
import copy
import pdb

class target:
    def __init__(self, label):
        self.id = str(label)
        self.source = ''

    def load_data_tess(self,datafile,removebad = True,sector_label=''):
        '''
            This function opens a standard TESS LC downloaded from mast or elsewhere
            By default quality flags that are not zero (all good) get yeeted, set removebad=False to keep everything
        '''
        hdu = fits.open(datafile)
        self.rawdata = hdu[1].data.copy()
        hdu.close()
        self.sourcetype = 'TESS'
        self.sourcefile = str(datafile)
        ##process to a more useful format:
        self.data = lcfunctions.LCconvertCTL(self.rawdata,removebad=removebad)
        self.sector=sector_label

    def load_data(self,time,flux,rawflux=[0],qual=[0],al=[0],source='user',sourcefile = 'user',sector_label='user',removebad=True):
        '''
            Load data from user input, maybe their own thing?
            By default quality flags that are not zero (all good) get yeeted, set removebad=False to keep everything

            INPUTS:
                time: time for each datapoint, in days
                flux: flux at each datapoint Should be normalized so that it sits at a baseline of 1, i.e. median divided would do it.
            OPTIONAL INPUTS:
                rawflux =[0]: A raw flux at each datapoint, default sets it all to zero
                qual=[0] : Quality flags as per tess or K2, default sets to zero meaning all fine
                al=[0]: arclength, useful for k2sff corrected K2, otherwise defaults to zero
                source = 'user': label for the source
                sourcefile ='user': If this data came from some file, save it here
                sector_label='user': Did it come from a particular k2 or tess sector? Put that here
        '''
        dl = len(time)
        outdata         = np.recarray((dl,),dtype=[('t',float),('fraw',float),('fcor',float),('s',float),('qual',int),('divisions',float)])
        outdata.t = time
        outdata.fcor=flux
        outdata.fraw[:]    = 0
        outdata.s[:]      = 0
        #outdata.detrend[:] = 0
        outdata.qual[:]    = 0
        if len(rawflux) == len(flux): outdata.fraw = rawflux
        if len(al)      == len(flux): outdata.s    = al
        if len(qual)    == len(flux): outdata.qual = qual
        okok = np.where(np.isnan(outdata.fcor)==False)[0]
        outdata.fraw /= np.nanmedian(outdata.fraw)
        outdata.fcor /= np.nanmedian(outdata.fcor)
        outdata   = outdata[okok]
        keep      = np.where(outdata.qual == 0)[0]

        if removebad == True:outdata   = outdata[keep]
        self.data = outdata.copy()
        outdata   = 0.0
        self.rawdata = 0
        self.sourcetype = str(source)
        self.sourcefile = str(sourcefile)
        self.sector=sector_label


    def load_data_k2(datafile):
        print('not implemented yet mate, use load_data and do it yourself!!!')

    def load_data_cpm(datafile):
        print('not implemented yet mate, use load_data and do it yourself!!!')

    def load_data_eleanor(datafile):
        print('not implemented yet mate, use load_data and do it yourself!!!')

    def plotlc(self,figsize=(10,5),returnfigs=False,alpha=0.6,justdata=False):
        '''
        A function to plot progress for this target variable, used for quick and interactive inspection
        Really a time saver for plotting as you go to sanity check things

        Will automatically plot detrended curves if found

        justdata=True will turn off the detrended curves
        returnfigs will make the code return: fig,ax
        alpha sets the overall alpha for matplotlib
        figsize lets you customize the figure size


        '''


        fig,ax = plt.subplots(figsize=figsize)
        ax.plot(self.data.t,self.data.fcor,'.',label='Corrected',zorder=100,alpha=alpha)
        ax.plot(self.data.t,self.data.fraw,'.',alpha=alpha)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Relative Brightness')

        if justdata==False:
            try:
                ax.plot(self.notch.t,self.notch.detrend,'.k',label='Notch detrended')
            except:
                dummy=0
            try:
                ax.plot(self.locor.t,self.locor.detrend,'.r',label='LOCoR detrended')
            except:
                dummy=0

        ax.legend()

        if returnfigs == True: return fig,ax


    def run_notch(self,window=0.5,mindbic=-1.0,useraw=False, verbose=False, show_progress=False):
        '''
        A wrapper to run the notch filtering pipeline user can specify
        window size, and also notch evidence strength (Delta Bayesian
        Information Criterion between notch and no-notch models) to accept
        notch.  The above defaults are usually fine for most things.
        '''


        notch = lcfunctions.run_notch(
            self.data, window=window, mindbic=mindbic, useraw=useraw, verbose=verbose, show_progress=show_progress
        )
        self.notch = notch.copy()
        self.notch_windowsize = window*1.0
        self.notch_mindbic = mindbic*1.0
        notch=0


    def run_locor(self,prot=None,alias_num=0.1,useraw=False):
        '''
        A wrapper to run the LOCoR pipeline

        User has to give a prot either as an input optional argument, or by
        making sure self has a prot attribute (self.prot = something)

        User can also specify the minimum period to alias the rotation
        period up to, to ensure data volume in each rotation is sufficient.

        The variable "alias_num" is the minimum rotation period to
        bunch rotations together to run LOCoR.  This is useful when your
        rotation period is really small and there's not enough points in
        each rotation. For TESS short cadence, this usually doesn't
        matter. At say 30 min cadence, a 0.5 day rotation
        period has only 24 points -> you might want to use alias_num=2.0 days
        to get ~100 points grouped together.
        '''

        if not prot:
            if hasattr(self,'prot') == False:
                print("You haven't provided a rotation period mate!")
                print("either call target.run_locor(prot=X) in days, or "
                      "do target.prot = X (in days) first then call "
                      "target.run_locor()")
                print("Also, remember to provide a period to alias to, "
                      "this is important for long cadence [Use alias_num=X "
                      "days argument], for short cadence TESS maybe 0.1 is "
                      "fine")
                return
            else: prot = self.prot

        locor = lcfunctions.run_locor(
            self.data, prot, alias_num=alias_num, useraw=useraw
        )

        self.locor = locor.copy()
        self.locor_prot = prot*1.0
        self.locor_alias = alias_num*1.0
        locor=0



    def combine_lc(self,newtarg,tocombine=['']):
        ##combines lightcurves
        ##Joins input data, then will by default search for like detrending modes (notch/locor) and join them.
        ## if tocombine is specified (e.g. tocombine=['notch','locor'] it will combine those detrends from self and newtarg respectively,
        ##best to only do this if you are a super user as details of logging extractions modes are not maintain well.

        if newtarg.id != self.id:
            print('Warning: These targets you want to join have different names!')



        ##start by combining the normal data pre-detrend and adjusting labels
        ctarg = copy.deepcopy(self)
        cdata = lcfunctions.LCcombine((self.data,newtarg.data))
        ctarg.data=cdata
        ctarg.sector = self.sector+'+'+newtarg.sector

        ##If user hasn't given instructions, combine like things
        if tocombine[0]=='':
            if hasattr(self,'notch') and hasattr(newtarg,'notch'):
                print('Combining notch detrends')
                cnotch = lcfunctions.LCcombine((self.notch,newtarg.notch))
                ctarg.notch = cnotch
                ctarg.notch_windowsize = [self.notch_windowsize*1,newtarg.notch_windowsize*1]
                ctarg.notch_mindbic = [self.notch_mindbic*1,newtarg.notch_mindbic*1]
            elif hasattr(self,'locor') and hassattr(newtarg,'locor'):
                print('Combining locor detrends')
                cocor = lcfunctions.LCcombine((self.locor,newtarg.locor))
                ctarg.locor = clocor
                ctarg.locor_prot = [self.locor_prot*1,newtarg.locor_prot*1]
                ctarg.locor_alias = [self.locor_alias*1,newtarg.locor_alias*1]

            else:
                print('There are not any matching extractions to combine so I just combined the input data!')

        ##If user asks for combination of different things, do that then
        if tocombine[0] != '':
            if hasattr(self,tocombine[0]) == False:
                print('target 1 does not have this type of extraction to combine: ' + str(tocombine[0]))
                print('So I just combined the input data')

            elif hasattr(newtarg,tocombine[1]) == False:
                print('target 2 does not have this type of extraction to combine: ' + str(tocombine[1]))
                print('So I just combined the input data')

            else:
                print('Combining user specified detrends into attribute cdetrend')
                c1 = getattr(self,tocombine[0])
                c2 = getattr(newtarg,tocombine[1])
                combdetrend = lcfunctions.LCcombine((c1,c2))
                ctarg.cdetrend = combdetred

        return ctarg





