# Notch_and_LOCoR

This package hosts the Notch filter and LOCoR detrending algorithms for space-based lightcurve data from TESS and Kepler K2, from the ZEIT and THYME papers,
developed by Aaron Rizzuto in https://arxiv.org/abs/1709.09670.   
   
To get started, clone the repo and run through the Quick_Start_Guide jupyter notebook!

If you want to combine lightcurves from multiple sectors/missions, take a look at TOI 2048.ipynb after going through Quick_Start_Guide.ipynb.

If you're interested in injection-recovery testing of lightcurves, check out Example_Injection_Recovery.ipynb and ijtesting.py, though this will require some tweaking of settings once you're ready to run it on a supercomputer.

Finally, if you want to use Notch and/or LOCoR as a base for your own ideas, take a look in lcfunctions.py to see how to directly access the core processes without the interface.py front end formats.


See these other papers for use cases and examples of planets found and other uses:<br>
https://arxiv.org/abs/1808.07068<br>
https://arxiv.org/pdf/1906.10703<br>
https://arxiv.org/abs/2005.00013
