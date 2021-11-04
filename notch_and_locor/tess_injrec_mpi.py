
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import sys,os
import time
import pdb
mpl.rcParams['lines.linewidth']   = 2
mpl.rcParams['axes.linewidth']    = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.labelsize']   = 10
mpl.rcParams['xtick.labelsize']   = 10
mpl.rcParams['axes.labelsize']    = 10
mpl.rcParams['legend.numpoints']  = 1
mpl.rcParams['axes.labelweight']  = 'semibold'
mpl.rcParams['axes.titlesize']    = 10
mpl.rcParams['axes.titleweight']  = 'semibold'
mpl.rcParams['font.weight']       = 'semibold'

## Import the required code from this package
import interface as thymeNL
import lcfunctions as lcfunc
import injection_recovery as IJ

##MPI stuff required for messaging between processes.
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

##Earth radius in Solar units, close enough to be useable.
rearth = 0.009154


'''
This script runs a large injection recovery test over a grid of planet parameters for a given star.
Currently setup for HIP67522 but is completely general otherwise.

Requires a working version of Message Passing Interface (MPI). I suggest OPENMPI
As well as the python package MPI4Py. Most of this will work off the bat on a supercomputing facility, but can also be run on any
machine with multiple cores/threads.

On a unix system with openmpi installed, use the following to call the script:

mpiexec -n N python tess_injrec_mpi.py

where N is the number of processes you want to involve.
For bugshooting use N=1 to start with

'''

### Below we specify the target and give the Stellar mass/radius for use later.
target_name = 166527623
datafile = 'tessdata/tess2019112060037-s0011-0000000166527623-0143-s_lc.fits'



mstar = 1.22
rstar = 1.38
thisprot  = 1.41
forcewindowsize = 0.5 #(set to a negative to let code decide)


##planet radius (rearth) and period (day) bounds for the grid. Currently uniformly distribbutes phase and sets ecc to zero
rlow  = 0.5
rhigh = 10.0
plow  = 1.00
phigh = 20.0
##define the MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
ncor = size


##Define the base directory, might have to change this on a supercomputer

basedir = './'
outdir = 'tessinjrec/'
logdir = 'logfiles/'
##notch settings
demode = 1 ##2 here calls locor, but note that requires a rotation period
forcenull=False
#import pdb
#pdb.set_trace()
jobid        = str(target_name)
datestamp    = time.strftime('%Y%m%d-%H:%M:%S', time.localtime(time.time())) ##output with starting timestamp so all runs are distinguishable

if rank == 0:
    if demode ==1 : print('Mode is Notch')
    if demode ==2 : print('Mode is LOCoR')
    if (demode != 1) & (demode != 2) :
        print('detrend mode (demode) set incorrectly')
        comm.Abort() ##This ends the mpi processes


##Determine how many test each core is doing based on how many test in total are wanted per star
npl = 2##1008 is divisible by a multiple of 24 so cores arent wasted, if you set this to 1 it will fail later,

##figure out tests per process
s   = npl/ncor
nx  = (s+1)*ncor - npl
ny  = npl - s*ncor
if rank <= nx-1: mynumber = int(s)
if rank > nx-1: mynumber = int(s)+1

##now it's clear how many test are running on each processor, tell the user
if rank == 0: print('Using ' + str(nx) + ' cores of size ' + str(s) + ' and ' +str(ny) + ' cores of size ' + str(s+1))

##identification string for later, so that processes have their own working files
ids = str(rank)

if rank == 0: print('Preprocessing Complete, Beginning......')
comm.barrier()

##now create an output directory for the results, should be unique:
if rank == 0:
    outdirname = basedir+outdir + jobid+ '_' + str(demode) + '_'+datestamp

    while os.path.exists(outdirname):##keep adding more stuff to name until directory is uniquem shouldn;'t happen at all as far as I can tell
        outdirname += '_1'
    os.makedirs(outdirname)
    ##open a log file for writing information:
    ijlog = open(outdirname +'/ijlogger.txt','wb')


ticnum = target_name ##The stars information and params
ticname = ticnum
sector = 11
sect = 'S11'



if demode == 2:
    windowsize = thisprot*1.0
if demode == 1:
    windowsize=1.0
    if thisprot < 2.0: windowsize=0.5
    if forcewindowsize > 0: windowsize=forcewindowsize *1.0

if rank == 0 :print('mass/rad/win ', mstar,rstar,windowsize)

if (mstar > 0.0) & (rstar > 0.0) & (windowsize > 0.0): ##Make sure we have a mass,radius and windowsize estimate

    ##only happens once per epic
    ##probably should only be done by one core then broadcasted out to the others:
    if rank == 0:
        hip67522 = thymeNL.target(target_name)
        hip67522.load_data_tess(datafile)
        rawdata = rawdata = hip67522.data
    else:rawdata=None
    if ncor > 1: rawdata = comm.bcast(rawdata,root=0)

    ##generate mynumber random sample planets orbits, logarithmic in planet radius
    pinj_vect = np.random.uniform(low=plow,high=phigh,size=mynumber)
    #rpinj_vect  = 10**np.random.uniform(low=np.log10(1.),high=np.log10(12.0),size=mynumber) ##this in in earth radii, which is what injrec_list_mpi takes
    ##a more complicated distribution to sample, half normal, with half beta-2/6, this puts the meat of the distribution around 2-4 R-earth, which is the key area for most systems
    ttt = np.mod(rank,2)
    ##because we are doing half/half normal and beta, decide what to do if number of samples is odd.
    if ttt == 1:
        nnn1 = int(np.ceil(mynumber/2.))
        nnn2 = mynumber/2
    else:
        nnn2 = int(np.ceil(mynumber/2.))
        nnn1 = int(mynumber/2)
    #import pdb
    #pdb.set_trace()
    rpinj_vect = np.concatenate((np.random.uniform(low=rlow,high=rhigh,size=nnn1),np.random.beta(2,6,size=nnn2)*(rhigh-rlow)+rlow))##earth radii
    phase_vect = np.random.uniform(low=0.0,high=1.0,size=mynumber)
    binj_vect = np.random.uniform(low=0.0,high=1.0,size=mynumber)
    einj_vect = np.zeros(mynumber,dtype=float)
    winj_vect = np.zeros(mynumber,dtype=float)

    ##turn it all into a list of points, with final parameter being the detection flag
    ##note this list is process dependent
    ijlist    = np.transpose(np.array([pinj_vect,rpinj_vect,binj_vect,phase_vect,einj_vect,winj_vect,pinj_vect*0.0,pinj_vect*0.0,pinj_vect*0.0,pinj_vect*0.0]))

    ##when testing using a normal computer, put a particular planet setup here
    #ijlist[0] = np.array([2.86677240736 ,8.58210005063 ,0 ,0.201926319663, 0.0 ,0.0 ,0.0,0.0,0.0,0.0])
    #ijlist = ijlist[0:1]
    #print ijlist.shape
    #injrec_tesslist_mpi(epic,pointlist,rawdata,mstar,rstar,ids,thisrank=-1,machinename='laptop',jobname='',
    #                wsize=1.0,demode=1,forcenull=False,alias_num=2.0,min_period=1.0001,max_period=12.0):

    ##run the injection-recovery on the local grid, ecode of -1 is a fail
    ij_result,ecode = IJ.injrec_tesslist_mpi(ticname,ijlist,rawdata,mstar,rstar,ids,thisrank=-1,
                    wsize=windowsize,demode=demode,forcenull=forcenull,alias_num=0.5,min_period=1.0001,max_period=20)
    print('one search done succesfully')
    if ecode == -1: ##if an exception got caught, output the error message and bail
        exceptfile = open(basedir + outdir + logdir/'exceptfile_' + str(rank) + '.txt','ab')
        exceptfile.write('!!!Error in injrec_list_mpi, Aborting, '+jobid+', '+time.strftime('%Y%m%d-%H:%M:%S', time.localtime(time.time())) +' \n')
        exceptfile.write( '!!!Except Info: ' + ij_result) ##this should be a string with information in the case of an exception
        exceptfile.flush()
        exceptfile.close()
        comm.Abort()

    ###again, for testing on desktops, print the output and kill the process
    #print ij_result
    #comm.Abort()

    ##now collate results from all the processors, the root (rank=0) process can do the collecting
    ##this is just stacking all the result lists from everyone
    if rank == 0:
        finalresult = ij_result.copy() ##initialize the result list from it's own results
        ##run through the number of processes and collate the output
        for nn in range(1,size):
            newstuff    = comm.recv(source=ANY_SOURCE)
            finalresult = np.append(finalresult,newstuff,axis=0)

    ##if not root process, just send your result list
    else:
        comm.send((ij_result),0)

    ##if you are root process and everything is finished, output a pkl file of the result list
    comm.barrier()
    if comm.rank == 0:
        print('Outputing result inj-rec grid (rank)',comm.rank)
        ##output the list too, just incase something is screwed in the grid
        outputter = open(outdirname + '/tic'+str(ticname)+'_ijmc_'+jobid+'_'+datestamp+'_'+str(windowsize)+'.pkl','wb')
        pickle.dump((finalresult,ticname,sector),outputter)
        outputter.close()
        ##now write to the log file
        #ijlog.write(str(thisepic) +' '  + str(nd) +' ' + str(ndf) +' ' + str(gradmode) +' ' + str(windowsize) +' ' + str(demode) + ' ' + str(mstar) + ' ' + str(rstar) +' \n')
        #ijlog.flush()

    #f comm.rank == 0: print 'Finished star ' + str(eee+1) + ' out of ' +str(len(epiclist)),comm.rank
    comm.barrier()

##end if statement for the case that we known the mass/radius/wsize to use
#else: ##output a message that this star was skipped due to lack of parameters
#    if comm.rank == 0: print 'Skipping star ' + str(eee+1)+ ' out of ' + str(len(epiclist)) + ' (no params)', comm.rank
#comm.barrier() ##make sure all the processes have caught up to each other so the results grid doesn't get overwritten?


##if everything is finished, abort
comm.barrier()
if rank == 0:
    ijlog.close()
    print('All Done')
    comm.Abort()
