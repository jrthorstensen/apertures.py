#!/usr/bin/env python3
"""apertures.py -- Python code to replace apedit and aptrace.  
     Starting with proof-of-concept.
"""

import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import numpy as np
import numpy.polynomial.legendre as leg
import argparse
import sys
import datetime # for timestamp
import os

########################################################
# 
#  Originally by John Thorstensen, Dartmouth College;
#  first version is from 2021 August.
#
# apertures.py -- Standalone python 3 program to replace
#   iraf 'apedit' and 'aptrace', for spectra of single
#   stellar targets.
# 
#   'apedit' and 'aptrace' define software apertures for
#   extracting the 1-d target spectra from the 2-d images;
#   they do not do the extraction.  To extract the 1-d 
#   spectra you can use either the iraf 'apsum' task or
#   my 'opextract.py'.
#
#   Limitations:
#
#      - Assumes there is only one target.
#      - Assumes the target is reliably centered somewhere
#          in a modest range of pixels along the slit.
#      - While not a formal requirement, it'll probably work 
#          best with point sources.
#      - Does all curve fits with legendre polynomials
#      - At present, the 'aptrace' step does not have any
#          provision for user input.
#
#   How to use it:
#
#      - Copy 'apertures.config' into the directories 
#          where your images reside.  This has adjustable  
#          parameters similar to the parameter files in 
#          iraf/pyraf.  The format should be pretty 
#          much self-explanatory. In what follows, parameters
#          from this file are listed in UPPERCASE.
#
#      - "apertures.py -h" gives the command line arguments.
#          The single positional argument is the image(s) 
#          you want to work on.  This can be a single image,
#          or a list of 2-d images, with or without the '.fits'
#          extension; if it is a list, prepend the name of the 
#          list with '@', just like in iraf.
#
#                apertures.py  myimage
#                apertures.py  @myimageist
#
#          Note especially the '-c' command line argument, that
#          specifies that dispersion runs along columns rather 
#          that along rows.
#      -  The program will compute an average profile around
#          DISPLINE, which you can choose to be at a wavelength
#          where you expect your sources to be bright.  The number
#          of individual lines averaged is NAVG, and the average
#          is a median to suppress cosmic rays.
#      -  A window of width PROFWIN centered on PROFCTR is searched
#          for the maximum.
#      -  The background is computed within limits B1 and B2 on the 
#          left, and B3 and B4 on the right; these are all in pixes
#          relative to the peak.  The fit is iterated several times
#          which generally removes other stars, glitches, etc.
#          The background fit parameters can be changed in the config
#          file but not interactively, but the limits can be changed 
#          on the fly (more later). 
#      -  A subarray is constructed from a range of pixels of width
#           WIDTH around the peak, and the background fit is subtracted
#           from this.
#      -  The aperture limits are found by 'walking' down from the 
#           peak of the background-subtracted subarray on either side
#           of the peak until a value is found that's less than 0.3 times 
#           the peak; the limit on either side is computed by 'growing'
#           the width on either side by a factor of 1.5.  This is very
#           similar to the iraf algorithm.
#      -  You can now adjust aperture using the following cursor commands;
#           to make an adjustment, set the cursor to the desired spot and type
#           the appropriate character.  (This is just like iraf).
#             - 1, 2, 3, and 4 -- set background limits and recompute.
#             - c -- recenter the aperture on the cursor.
#             - [ and ] -- change the left or right aperture limit.
#             - ? -- print a menu.
#             - q -- quit (this is a matplotlib default).
#           The graph is redrawn after each command.
#      - Note that the points used in the background fit are overplotted
#         in red; blue points have been rejected by the iterations.
#      - At present I haven't implemented commands for re-setting the 
#         plot limits.  You can use matplotlib's 'magnifying glass'
#         to blow up a region of interest, and the little 'house' icon
#         to return to the original scale.
#      
#      After the aperture is defined, the program does an equivalent of
#        'aptrace' -- it runs up and down the dispersion and finds the 
#         aperture center at each wavelength.  It then fits a polynomial
#         of order TRACEORD to the points, and iterates TRACEITER times  
#         eliminating points outside of TRACECLIP sigma.  
#
#      The trace is presented in a plot, with (again) the points used
#         overplotted in red.  One can adjust the fit:
#             - d  Delete the point nearest the cursor and refit.
#             - r  Restore the point nearest the cursor and refit.
#             - R  Restore ALL the deleted points, and refit.
#             - )  Delete ALL points to the RIGHT of the cursor and refit.
#             - (  Delete ALL points to the LEFT  of the cursor and refit.
#             - o or O (lower and upper case) respectively decrease or
#                 increase the order of the fitted polynomial by 1.
#             - i  iterate the fit the default number of times and 
#                   reject discrepant pixels at each iteration.
#             - ?  Print a menu.
#             - q  Quit the fitting and proceed.
#        The fit is recomputed and replotted after each command.
#        NOTE that with d, r, R, ), and (, no further points are rejected.
#              but with o, O, and i, the iteration/rejection cycle is run.
#
#      When you're done with the fitting, the program writes out the aperture
#        information.  The fit is written to a file named 'ap' + (image root),
#        e.g. apccd123 for image 'ccd123', by default in the 'database' 
#        subdirectory under your current directory; the subdirectory is created if it
#        doesn't exist.  Any previous 'ap' file is overwritten.
#  
#      The format of the database 'ap' file is designed to be identical
#        to the format used by iraf.  This is understood by my 'opextract.py'
#        program.  Either iraf's 'apsum' or my own program can now be used to
#        extract the spectrum.   
#
#    If you're doing an '@' list, the program moves on to the next object. 
#
############################################################################

def legoverdomain(xarr, yarr, domainends, order) : 
    # the numpy 'legfit' just fits the values.  We want to write
    # a output that's compatible with input to IRAF, which 
    # wants the ends of the domain and legendre coeffs that
    # assume the x-values have been rescaled to [-1,1]

    # This takes an array, and a tuple giving [xmin,xmax]
    # and does the fit.
    # I'm passing in the domain limits separately since the 
    # xarr is often truncated by iterations.
    
    # order is the standard order (e.g. order 2 = parabolic).

    # recenter on mean of domainends


    xarr = xarr - (domainends[1] + domainends[0]) / 2.
    # and rescale.
    xarr = xarr * 2. / (domainends[1] - domainends[0]) 
    
    scaledcoeffs = leg.legfit(xarr,yarr,order) 

    return scaledcoeffs

def evaluate_legdomain(xarr, scaledcoeffs, domainends) :

    # numpy's Legendre object can be set up with scaled coeffs 
    # and a domain

    lpoly = leg.Legendre(scaledcoeffs, domain = domainends)  
    return lpoly(xarr)

def write_iraf_legendre(imname, apcenter, lowap, highap, 
    b1rel, b2rel, b3rel, b4rel, coeffs, path = './database') :
    
    # Writes an 'aperture' database file in a form compatible 
    # with IRAF.  Arguments are the name of the image (used
    # for constructing the file name), the four background limits
    # relative to the aperture center, the lower and upper limits of
    # the data aperture, the legendre coefficients giving the 
    # aperture trace (position vs. dispersion), the domain of the
    # legendre (generally the length of the spec in pixels), and
    # an optional path for the file to be written to; defaults
    # to the iraf standard './database'.  

    # This is coded to extract only one target.

    # print(f"path is {path}")

    if not os.path.exists(path) :
        try :
             os.makedirs(path)
        except :
             print(f"Could not make the directory {path}!")

    if '.fits' in imname :
        imname = imname.replace('.fits','')

    if path[-1] == '/' :
        outfname = path + 'ap' + imname
    else : 
        outfname = path + '/ap' + imname
    
    timenow = datetime.datetime.now()
    # using y-m-d, inverting ancient IRAF format.
    timestring = timenow.strftime("%a %H:%M:%S %Y-%b-%d")

    # print(f"{timestring} {outfname}")
  
    outf = open(outfname,"w")
    
    outf.write(f"# {timestring}\n")
    dl = config['DISPLINE']
    # if dispersion is along columns, the numbers change oddly.
    if args.columndispers : 
        outf.write(f"begin\taperture {imname} 1 {apcenter:7.3f} {data.shape[1] - dl}\n")
    else :
        outf.write(f"begin\taperture {imname} 1 {dl:5.0f} {apcenter:7.3f}\n")
    # The leading '\t' has to be stripped out because it somehow breaks the 
    # f string and throws an error.
    outf.write("\t" + f"image\t{imname}\n")
    outf.write("\t" + f"aperture\t1\n")
    outf.write("\t" + f"beam\t1\n")
    if args.columndispers :  # again, column dispersion changes order of things.
        outf.write("\t" + f"center\t{apcenter:7.3f} {data.shape[1] - dl:5.0f}\n")
        outf.write("\t" + f"low\t{lowap - apcenter : 6.2f} {-1. * (data.shape[1] - dl) : 5.0f} \n")
        outf.write("\t" + f"high\t{highap - apcenter : 6.2f} {dl}\n")
    else :
        outf.write("\t" + f"center\t{dl:5.0f} {apcenter:7.3f}\n")
        outf.write("\t" + f"low\t{-1. * dl + 1 : 5.0f} {lowap - apcenter : 6.2f}\n")
        outf.write("\t" + f"high\t{data.shape[1] - dl} {highap - apcenter : 6.2f}\n")
    outf.write("\tbackground\n")
    outf.write("\t\t" + f"xmin {b1rel}\n")
    outf.write("\t\t" + f"xmax {b4rel}\n")
    outf.write("\t\t" + f"function legendre\n")
    outf.write("\t\t" + f"order {nord + 1}\n")  # iraforder.
    outf.write("\t\t" + f"sample {b1rel}:{b2rel},{b3rel}:{b4rel} \n")
    outf.write("\t\t" + f"naverage 1\n")
    outf.write("\t\t" + f"niterate {niterate} \n")
    outf.write("\t\t" + f"low_reject {clip} \n")
    outf.write("\t\t" + f"high_reject {clip}\n")
    outf.write("\t\t" + f"grow 0\n")
    if args.columndispers :
       outf.write("\t" + f"axis 1\n")
    else :
       outf.write("\t" + f"axis 2\n")
    ncurve = 4 + len(coeffs) 
    outf.write("\t" + f"curve\t{ncurve}\n")
    outf.write("\t\t2.\n")  # type is always legendre (type 2)
    outf.write("\t\t" + f"{len(coeffs)}\n") # iraf 'order' is number of coeffs.
    outf.write("\t\t1.\n") # trace fit domain always starts in col. 1
    outf.write("\t\t" + f"{float(data.shape[1]-1)}\n")
    outf.write("\t\t" + f"{coeffs[0] - apcenter}\n")   # iraf writes zeroth coeff
          # with starting point subtracted.
    for coeffnum in range(1,len(coeffs)) :
        outf.write("\t\t" + f"{coeffs[coeffnum]}\n")
    outf.write("\n")  

    outf.close()

def getconfig(infname) :
    # read in the parameters from the config file.

    # Format is  PARAM | value | datatype

    try : 
        inf = open(infname,'r')
    except : 
        print(f"Could not open config file {infname} - dying.")
        sys.exit()
    configdict = {}
    for l in inf : 
        x = l.split('#')
        if '|' in x[0] :
            y = x[0].split('|')
            if 'int' in y[2].lower() : 
                configdict[y[0].strip()] = int(y[1].strip())
            elif 'float' in y[2].lower() : 
                configdict[y[0].strip()] = float(y[1].strip())
            else : 
                configdict[y[0].strip()] = y[1].strip()

    inf.close()
    return configdict

def getimlist(instring) :

    # interprets the positional argument.  If it's a list,
    # prepend the name with "@"; otherwise it will look for a
    # single file of the specified name.  Files can be specified with or
    # without the '.fits' suffix, but the actual files need to
    # have the '.fits' suffix to be found.

    frootlist = []
    imnamelist = []

    if instring[0] == '@' :
        listfname = instring[1:]
        try :
            listf = open(listfname, 'r')
        except :
            print(f"image list {listfname} did not open. Dying.")
            sys.exit()
        for l in listf :
            l = l.strip()
            if len(l) > 0 : 
                if '.fits' in l:
                    frootlist.append(l.replace('.fits',''))
                    imnamelist.append(l)
                else :
                    frootlist.append(l)
                    imnamelist.append(l + '.fits')

    else :
        if '.fits' in instring :
            imnamelist.append(instring)
            frootlist.append(instring.replace('.fits',''))
        else :
            imnamelist.append(instring + '.fits')
            frootlist.append(instring)

    return(frootlist, imnamelist) 

def avgcols(data,midcol,navg) :

    # returns an array of 1-indexed row numbers and the
    # median over the specified columns
    # [dispersion is along rows.  Array is transposed
    # before this if needed.]

    xdisp = np.arange(1,data.shape[0]+1) # 1-index

    # check bounds

    if midcol < data.shape[1] : 
        startcol = int(midcol - navg / 2)
        if startcol < 0 : startcol = 0
        endcol = int(midcol + navg / 2)
        if endcol > data.shape[1] : endcol = data.shape[1] - 1
        # print(data[0:data.shape[0],startcol:endcol].shape)

        return (xdisp,np.median(data[0:data.shape[0], startcol:endcol,],axis = 1))

    else:
        print("column to average is not witin bounds.")
        return (xdisp,np.zeros(data.shape[0]))

def fitsky(xrows, profile, b1, b2, b3, b4, nord, niterate, clip, 
    doplot=True, verbose = True) :

    global bckgpts
    global bckgfitline

    # xrows   : array of 1-indexed pixel numbers
    # profile : array of medians (cross-dispersion) of a slice 
    # b1, b2  : pixel limits of left bckgrd region in chip coords
    # b3, b4  : pixel limits of right bckgrd region in chip coords
    # nord    : order of the legendre fit to background
    # niter   : number of rejection iterations
    # clip    : sigma to clip at each iteration

    # Build arrays to fit.

    leftbackx = xrows[b1:b2]
    leftbacky = profile[b1:b2]
    rightbackx = xrows[b3:b4]
    rightbacky = profile[b3:b4]

    xarr = np.hstack((leftbackx,rightbackx))
    yarr = np.hstack((leftbacky,rightbacky))

    for niter in range(0, niterate) :
        legcoefs = leg.legfit(xarr,yarr,nord)
        fit = leg.legval(xarr,legcoefs)
        residuals = yarr - fit
        stdev = np.std(residuals) 
        if stdev < 0.1 :
            print("Breaking - stdev < 0.1")
            break;
        keepindices = abs(residuals) <= clip * stdev
        xarr = xarr[keepindices]
        yarr = yarr[keepindices]
        if verbose : print(f"bckg iteration {niter}; {len(xarr)} points.")

    bckgatctr = leg.legval(int(indmax),legcoefs)
    if verbose : print(f"bckgatctr {bckgatctr}")

    bckgfit = leg.legval(xrows,legcoefs)
    if doplot : 
        if bckgfitline is not None : 
            bfitl = bckgfitline.pop(0) # it's a tuple
            bfitl.remove()
        bckgfitline = plt.plot(xrows, bckgfit) 
    if doplot : 
        # erase old points if any
        if bckgpts is not None : 
            bpts = bckgpts.pop(0) # it's a tuple
            bpts.remove()
        bckgpts = plt.plot(xarr, yarr,'ro',markersize=3)

    return (bckgfit, bckgatctr) 

def fittrace(traceord, traceiter, doplot = True,
    verbose = False) :

    global colpts, colfitline
    global columns, profctrs

    # print(f"colpts {colpts} colfitline {colfitline}")

    # do one fit outside the loop if traceiter is 0.

    if traceiter == 0 :
        legcoefs = leg.legfit(columns, profctrs, traceord)
        fit = leg.legval(columns,legcoefs)
        residuals = profctrs - fit
        stdev = np.std(residuals) 
    
    else :
        for i in range(0,traceiter) :
    
            # print(f"traceord = {traceord} len(columns) = {len(columns)}")
    
            legcoefs = leg.legfit(columns, profctrs, traceord)
            fit = leg.legval(columns,legcoefs)
            residuals = profctrs - fit
            stdev = np.std(residuals) 
    
            if stdev < 0.001 :
                print("Breaking -  stdev < 0.001")
                break
            else :
                keepindices = abs(residuals) <= traceclip * stdev
                #print(keepindices)
                #print(columns)
                #print(profctrs)
                columns = columns[keepindices]
                profctrs = profctrs[keepindices]
                if verbose : 
                    print(f"trace iteration {i}; {len(columns)} points.")
        
    scaledcoeffs =  legoverdomain(columns, profctrs, (1, data.shape[1] - 1), traceord)  
    # print(f"scaled coeffs: {scaledcoeffs}")
    fit = evaluate_legdomain(origcolumns, scaledcoeffs, (1, data.shape[1] - 1))
    # print(f"fit = {fit}")

    if doplot : 
        if colfitline is not None :  # delete old artists
            # print("removing colfitline.")
            colfitl = colfitline.pop(0) # it's a tuple
            colfitl.remove()
        if colpts is not None :
            colptsl = colpts.pop(0)
            # print(f"colptsl = {colptsl}") 
            colptsl.remove()
        # print("plotting.")
        colfitline = plt.plot(origcolumns,fit) 
        # print(f"len(profctrs) {len(profctrs)}")
        colpts = plt.plot(columns, profctrs, 'ro',markersize=3)
        plt.title(f"{im} {hdr['object']} - keeping {len(columns)}. order {traceord}, iterated {traceiter}, stdev {stdev: 5.2f}")
        
    return scaledcoeffs
 
def onpress(event) :  # This does all the human interaction for the
    # aperture plot.

    global bckgpts
    global bckgfitline
    global b1, b2, b3, b4
    global centerline, lowapline, highapline
    global apcenter, aplow, aphigh

    # Use 1, 2, 3, or 4 to change background limits and replot.

    if event.key == '1' :
        b1 = int(event.xdata)
        (bckgfit, bckgatctr) = fitsky(xrows, profile, b1, b2, b3, b4, nord,
            niterate, clip, verbose = False) 
        plt.draw()
    if event.key == '2' :
        b2 = int(event.xdata)
        (bckgfit, bckgatctr) = fitsky(xrows, profile, b1, b2, b3, b4, nord,
            niterate, clip, verbose = False) 
        plt.draw()
    if event.key == '3' :
        b3 = int(event.xdata)
        (bckgfit, bckgatctr) = fitsky(xrows, profile, b1, b2, b3, b4, nord,
            niterate, clip, verbose = False) 
        plt.draw()
    if event.key == '4' :
        b4 = int(event.xdata)
        (bckgfit, bckgatctr) = fitsky(xrows, profile, b1, b2, b3, b4, nord,
            niterate, clip, verbose = False) 
        plt.draw()
    if event.key == 'c' :
        apcenter = event.xdata
        centerline.remove()
        centerline = plt.axvline(apcenter)
        plt.draw()
    if event.key == '[' :
        aplow = event.xdata
        lowapline.remove()
        lowapline = plt.axvline(aplow, color='green', linestyle = 'dashed')
        plt.draw()
    if event.key == ']' :
        aphigh = event.xdata
        highapline.remove()
        highapline = plt.axvline(aphigh, color='green',linestyle = 'dashed')
        plt.draw()
    if event.key == 'p' :
        print(f"[low : center : high] = [{aplow:5.1f} : {apcenter:5.1f} : {aphigh:5.1f}]")
        print(f"background: [{b1:4.0f}:{b2:4.0f},{b3:4.0f}:{b4:4.0f}]")
    if event.key == '?' :
        print("Aperture finding key commands:")
        print("c - Center aperture on cursor.")
        print("[ - Set lower aperture limit.")
        print("] - Set upper aperture limit.")
        print("1 - set lower left background limit")  
        print("2 - set lower right background limit")  
        print("3 - set upper left background limit")  
        print("4 - set upper right background limit")  
        print("p - print current paramters.")
        print("q - accept fit and quit.")
  #  if event.key == 'q' :
  #      print('got q.')

def findnearest(x, y, xpts, ypts, xspan, yspan) : 

    # picks the point among (xpts, ypts) nearest to the given 
    # x and y (from the cursor) on a graph with ranges
    # xspan and yspan.  I know point-picking is available with 
    # the mouse but I will also need the character typed.

    # spans are needed because the scales of the axes are
    # often wildly different, so want to convert to something 
    # like the distance on the screen.  The graph is generally
    # not square, but this is close enough.

    # xpts and ypts are numpy arrays, so can vectorize.

    dxnorm = (xpts - x) / xspan
    dynorm = (ypts - y) / yspan
  
    # reference is Pythagoras (-500); no need to take 
    # sqrt since we just want the minimum index.

    misses = dxnorm * dxnorm + dynorm * dynorm  
  
    return np.argmin(misses)     

def onpress2(event) :  # This does the human interaction for the trace.

    global columns, profctrs, traceord
    global scaledcoeffs

    # 'lines' that may need to be deleted to revise the plot
    global colpts
    global tracefitline
    
    if event.key == 'O' :
        traceord = traceord + 1
        scaledcoeffs = fittrace(traceord,traceiter) 
        plt.draw()

    if event.key == 'o' : 
        if traceord > 0 :
            traceord = traceord - 1
            scaledcoeffs = fittrace(traceord,traceiter) 
            plt.draw()
        else :
            print("Fit order can't be less than zero.")

    if event.key == 'd' :
        # delete point nearest to the cursor.
        #print(f"start: {len(profctrs)} points.")
        axes = plt.gca()
        xlims = axes.get_xlim()
        ylims = axes.get_ylim()
        killindex = findnearest(event.xdata, event.ydata, columns, profctrs, 
           xlims[1] - xlims[0], ylims[1] - ylims[0])
        columns = np.delete(columns, killindex)
        profctrs = np.delete(profctrs, killindex)
        #print(f"end: {len(profctrs)} points.")
        # print(f"picked index {killindex}; coords {columns[killindex]} {profctrs[killindex]}")
        
        # refit without iteration.
        scaledcoeffs = fittrace(traceord,0) 
        plt.draw()

    if event.key == 'r' :
        # restore point nearest to cursor.
        # print(f"start: {len(profctrs)} points.")
        axes = plt.gca()
        xlims = axes.get_xlim()
        ylims = axes.get_ylim()
        addindex = findnearest(event.xdata, event.ydata, origcolumns, origprofctrs, 
           xlims[1] - xlims[0], ylims[1] - ylims[0])
        # appending these means they won't be in order any more.  This may cause
        # issues with plotting; we'll see.
        # print(f"attempting to add pt at {origcolumns[addindex]} {origprofctrs[addindex]}")
        columns = np.append(columns, origcolumns[addindex]) 
        profctrs = np.append(profctrs, origprofctrs[addindex])
        # print(f"end: {len(profctrs)} points.")

        # refit without iteration, or else the point gets thrown out again!
        scaledcoeffs = fittrace(traceord,0) 
        plt.draw()

    if event.key == 'R' :
        # restore ALL points, fit without iteration.
        columns = origcolumns 
        profctrs = origprofctrs
        # print(f"end: {len(profctrs)} points.")

        # refit without iteration, or else the point gets thrown out again!
        scaledcoeffs = fittrace(traceord,0) 
        plt.draw()

    if event.key == 'i' :
        # just fit, with iteration.
        # print(f"end: {len(profctrs)} points.")

        # refit without iteration, or else the point gets thrown out again!
        scaledcoeffs = fittrace(traceord,traceiter) 
        plt.draw()

    if event.key == '(' :  # kill all points below cursor x-position
        killindices = []
        for ind, x in enumerate(columns) : 
            if x < event.xdata :
                killindices.append(ind) 
        columns = np.delete(columns, killindices) 
        profctrs = np.delete(profctrs, killindices) 
        scaledcoeffs = fittrace(traceord,0) 
        plt.draw()

    if event.key == ')' :  # kill all points above cursor x-position
        # collect indices of points to kill, and then do the all
        # at once; if you try to do them one-by-one it changes the 
        # indexing as you and you run off the end.
        killindices = []
        for ind, x in enumerate(columns) : 
            if x > event.xdata :
                killindices.append(ind) 
        columns = np.delete(columns, killindices) 
        profctrs = np.delete(profctrs, killindices) 
        scaledcoeffs = fittrace(traceord,0) 
        plt.draw()

    if event.key == '?' :
        print("Aperture tracing key commands:")
        print(" ")
        print("NOTE: the fit is redone after each command, without")
        print("iterative rejection, for 'd', 'r', 'R', ')', and '('"),
        print("and with iteration for 'i','o', and 'O'.")
        print(" ")
        print("d - delete nearest point from fit.")
        print("r - restore nearest point to fit.")
        print("R - restore ALL the points to the fit.")
        print("i - refit with iteration (e.g., after 'R')")
        print(") - delete all points at higher x-values.")
        print("( - delete all points at lower x-values.")
        print("o - (lower case) - decrease poly order by 1.")
        print("O - (upper case) - increase poly order by 1.")
        print("q - accept fit and quit.")

    # Note that the 'q' key relies on matplotib to up and quit the
    # loop.  It would be good to have an option to quit without
    # writing, but it isn't essential.
 
def ridgeline(data, startline, colwidth, startcenter, apoffset,
  aplow, aphigh, 
  nord, niterate, clip, verbose = False) : 

# walks up and down the dispersion and finds profile at each location
# non-interactively, i.e., the points to be fitted for the trace.
     
    colnum = startline
    lastctr = startcenter

    # Figure out relative aperture limits so that search window can
    # move with the spectrum.

    relaplow = aplow - lastctr 
    relaphigh = aphigh - startcenter 
    if verbose : print(f"relaplow high {relaplow} {relaphigh}") 

    ctrdict = {}  # to facilitate sorting at end.
    ctrdict[startline] = startcenter

    # step through columns, compute profile centers.

    while colnum < data.shape[1] :
       # colnum = colnum + colwidth  # Don't update colnum until the end -- this avoids
       # the need for the awkward 'break' statement and ensures that the computation is 
       # exactly the same for the displine as for all the others.
       # if colnum > data.shape[1] : break
       xrows, profile = avgcols(data,colnum,colwidth) 
       if verbose : print(f"colnum {colnum} lastctr {lastctr} aplow {aplow} aphigh {aphigh}")
       aplow = lastctr + relaplow
       aphigh = lastctr + relaphigh

       # The aperture center is computed as a first-moment.  This has the consequence that
       # at high S/N, if the aperture limits don't include every bit of flux, one will see slight
       # discontinuities in the aperture center when a new pixel is included as we step through
       # columns.  This should should have no appreciable effect.  If it is a problem, then 
       # either (a) write a more elaborate scheme that deals with fractional pixels at the edge,
       # or (b) find another centering algorithm, like the convolution methods used for line
       # centering.
 
       searchstart = int(aplow)
       searchend = int(aphigh)
       if verbose : print(f"searchstart {searchstart} searchend {searchend}")
       fmax = -10000000.
       indmax = 0
#       print("search :",searchstart,searchend)
       for i in range(searchstart,searchend) : 
           if profile[i] > fmax : 
              fmax = profile[i]
              indmax = xrows[i]  # not 'i' itself!

       # be sure to move background window with spec
       b1 = int(indmax + config['B1'])
       b2 = int(indmax + config['B2'])
       b3 = int(indmax + config['B3'])
       b4 = int(indmax + config['B4'])

       (bckgfit, bckgatctr) = fitsky(xrows, profile, b1, b2, b3, b4, nord,
           niterate, clip, verbose = False, doplot=False) 

       profile = profile - bckgatctr
       
       ctr = np.sum(xrows[searchstart:searchend] * profile[searchstart:searchend]) 
       ctr = ctr / np.sum(profile[searchstart:searchend]) 
       ctr = ctr + apoffset # if the user has reset the center interactively, offset it here, too.
       if verbose : print(f"ctr = {ctr}")
       # if it didn't even land in the window, just revert to last result
       if ctr < searchstart or ctr > searchend :  
           # looking at a number of these reveals that the profile is basically lost.
           # just punt and replace it with the expected value.
           ctr = lastctr
#           plt.clf()
#           plt.plot(xrows,profile)
#           plt.title('Bad profile centering example.')
#           plt.show()
       ctrdict[colnum] = ctr       
       lastctr = ctr
       colnum = colnum + colwidth  

    # step down through columns from start.  Lots of copypasta
    # but I'm too lazy to write a function.  Reset back to beginning:
              
    colnum = startline - colwidth        
    lastctr = startcenter

    while colnum > 0 :
       
       colnum = colnum - colwidth  
       if colnum < 0 : break

       xrows, profile = avgcols(data,colnum,colwidth) 
       aplow = lastctr + relaplow
       aphigh = lastctr + relaphigh
       searchstart = int(aplow)
       searchend = int(aphigh)
       fmax = -10000000.
       indmax = 0
#       print("search :",searchstart,searchend)
       for i in range(searchstart,searchend) : 
           if profile[i] > fmax : 
              fmax = profile[i]
              indmax = xrows[i]  # not 'i' itself!

       # be sure to move background window with spec
       b1 = int(indmax + config['B1'])
       b2 = int(indmax + config['B2'])
       b3 = int(indmax + config['B3'])
       b4 = int(indmax + config['B4'])

       (bckgfit, bckgatctr) = fitsky(xrows, profile, b1, b2, b3, b4, nord,
           niterate, clip, verbose = False, doplot=False) 

       profile = profile - bckgatctr
       
       ctr = np.sum(xrows[searchstart:searchend] * profile[searchstart:searchend]) 
       ctr = ctr / np.sum(profile[searchstart:searchend]) 
       ctr = ctr + apoffset # if the user has reset the center interactively, offset it here, too.
       # pathology - 
       if ctr < searchstart or ctr > searchend :  
           # looking at a number of these reveals that the profile is basically lost.
           # just punt and replace it with the expected value.
           ctr = lastctr

       ctrdict[colnum] = ctr       
       lastctr = ctr
 
       ctrdict[colnum] = ctr       
       lastctr = ctr

    columns = sorted(ctrdict)  # the keys in order
    # there's probably some fabulous pythonic way to do these next two steps
    profctrs = []
    for c in columns : profctrs.append(ctrdict[c])
    
    return(np.array(columns), np.array(profctrs)) 

##############################################
# MAIN main Main part of program starts here. 
##############################################

parser = argparse.ArgumentParser("Define and trace spectral apertures.")

parser.add_argument('imageroots')    # positional - single image name, or list of them.
parser.add_argument('-c','--columndispers',help='Dispersion is along columns',action='store_true')
parser.add_argument('-p','--paramfile',help='Use non-default config filename',type=str,default="apertures.config")
parser.add_argument('-d','--defaultcursor',help='Turn off the crosshair cursor',action='store_true')

args = parser.parse_args()

config = getconfig(args.paramfile)

frootlist, imlist = getimlist(args.imageroots) 

for im in imlist :

    hdu = fits.open(im)
    hdr = hdu[0].header
    data = hdu[0].data

    fig = plt.figure(figsize=(config['FIGWID'],config['FIGHGHT']))
    cid = fig.canvas.mpl_connect('key_press_event',onpress)

    # Transpose the data if dispersion runs along columns.
    if args.columndispers : 
        data = data.T    

    xrows, profile = avgcols(data,config['DISPLINE'],config['NAVG'])

    searchstart = int(config['PROFCTR'] - config['PROFWIN'] / 2.)
    searchend =   int(config['PROFCTR'] + config['PROFWIN'] / 2.)
    fmax = -10000000.
    indmax = 0
    # print("search :",searchstart,searchend)
    for i in range(searchstart,searchend) : 
        if profile[i] > fmax : 
           fmax = profile[i]
           indmax = xrows[i]  # not 'i' itself!

    # print("fmax: ",fmax)
    plt.plot(xrows,profile)
    plt.xlabel("Pixel across dispersion")
    plt.ylabel("Counts")

    bckrange = config['B4'] - config['B1']  # full width of background regions
    xlow = indmax + config['B1'] - 0.05 * bckrange
    xhigh = indmax + config['B4'] + 0.05 * bckrange

    # fit the background
    # note configs are relative to aperture.
    b1 = int(indmax + config['B1'])
    b2 = int(indmax + config['B2'])
    b3 = int(indmax + config['B3'])
    b4 = int(indmax + config['B4'])
    niterate = config['BITER']
    nord = config['BFITORD']
    clip = config['BSIG']
    traceord = config['TRACEORD']
    traceiter = config['TRACEITER'] 
    traceclip = config['TRACECLIP']

    # initialize these - they get stored when the plot is drawn,
    # and then erased if the user changes the limits etc.
 
    bckgpts = None  # line instance of dots showing points used for bckgrd.
    bckgfitline = None  # line instance for plotted background fit.

    (bckgfit, bckgatctr) = fitsky(xrows, profile, b1, b2, b3, b4, nord,
        niterate, clip, verbose = False) 

    # scale y axis
    yrange = fmax - bckgatctr
    ylow = bckgatctr - 0.15 * yrange
    yhigh = fmax + 0.05 * yrange

    lowapind = int(indmax - config['WIDTH'] / 2)
    highapind = int(indmax + config['WIDTH'] / 2)

    # centroid, first moment using vector operations.
    # apy will be a subarray containing only the region around 
    # the profile.
    apy = profile[lowapind:highapind] - bckgfit[lowapind:highapind]
    # print(f"apy: {apy}")
    wgt = apy * apy
    xsum = np.sum(xrows[lowapind:highapind] * wgt)
    wgtsum = np.sum(wgt) 
    xfirstmom = xsum / wgtsum
    apcenter = xfirstmom  # save the estimate.

    # print(f"first pass: search limits {lowapind} {highapind}, apcenter = {apcenter}")

    # print(f"xfirstmom {xfirstmom}")

# ********
    # Tried getting a FWHM from 2nd moment, but it wasn't reliable enough.

    # print(f"squared-subbed: {(xrows[lowapind:highapind] - xfirstmom) ** 2}")
 #   var = np.sum(apy * ((xrows[lowapind:highapind] - xfirstmom) ** 2))
    # print(f"var before div. {var}, np.sum(apy) {np.sum(apy)}")
 #   var = var / np.sum(apy)
    # print(f"var before sqrt. {var}")
 #   sig = np.sqrt(var)
    # print(f"sig: {sig}")
 #   fwhm = sig * 2.3  # about right for a gaussian
 #   lowap = xfirstmom - fwhm
 #   highap = xfirstmom + fwhm
# ********

    # try the iraf method - walk down from max to a specified fraction of the 
    # peak, and then 'grow' the limits by a specfied factor.

    fidlevel = 0.3    
    apgrow = 1.8

    apymaxind = np.argmax(apy)
    maxval = apy[apymaxind]
    crity = fidlevel * maxval  # level at desired 'fidlevel'

    # walk out from center index until less than crity:  

    # left side
    j = apymaxind
    while apy[j] > crity and j > 0 : j = j - 1
    # 'should' refine this with a straight-line fit, for now
    # just see if it works
    indhalfwid = apymaxind - j
    # indmax is relative to whole array, as is indlo. 
    indlo = indmax - apgrow * indhalfwid
    aplow = float(indlo) 

    # right side
    j = apymaxind
    while apy[j] > crity and j < len(apy) - 1 : j = j + 1
    indhalfwid = j - apymaxind
    indhi = indmax + apgrow * indhalfwid
    aphigh = float(indhi) 

    # recompute the first-moment center using the exact same 
    # limits as used in the ridgeline code.

    # print(f"indlo, indhi {indlo},{indhi}, types {type(indlo)}, {type(indhi)}")

    apy = profile[int(aplow):int(aphigh)] - bckgfit[int(aplow):int(aphigh)]
    # print(f"apy: {apy}")
    wgt = apy * apy
    xsum = np.sum(xrows[int(aplow):int(aphigh)] * wgt)
    wgtsum = np.sum(wgt) 
    xfirstmom = xsum / wgtsum
    apcenter = xfirstmom  # save the estimate.

    # print(f"second pass: search limits {int(aplow)} {int(aphigh)}, apcenter = {apcenter}")

    # save the line instances for replotting.
    centerline = plt.axvline(apcenter)

    # These show the extent of the window that's analyzed.

    plt.axvline(lowapind,color='black',linestyle = 'dotted')
    plt.axvline(highapind,color='black',linestyle = 'dotted')

    # These show the extent of the software aperture.

    lowapline = plt.axvline(aplow,color='green',linestyle='dashed')
    highapline = plt.axvline(aphigh,color='green',linestyle='dashed')

    plt.xlim(xlow,xhigh)
    plt.ylim(ylow,yhigh)
    plt.title(f"{im} {hdr['object']}")
    if not args.defaultcursor :
       axes = plt.gca()
       cursor = Cursor(axes,useblit=True,color='red',linewidth=1)
    plt.show()

    # The plt.show() sends over execution to the interactive 
    # aperture editing. When it's done, execution continues:

    # disconnect the 'cid', which seems to stop the 'event loop is running'
    # error message.
    fig.canvas.mpl_disconnect(cid)

    # if the aperture center has been offset by the user, we'll
    # want to offset all the automatically-determined center by the
    # same amount when we trace the aperture.  So safe this offset
    # and pass it to ridgeline.

    apoffset = apcenter - xfirstmom
    # And also keep the background limits relative to the aperture.
    b1rel = int(b1 - apcenter)
    b2rel = int(b2 - apcenter)
    b3rel = int(b3 - apcenter)
    b4rel = int(b4 - apcenter)

## Now that the limits for the aperture and background are found,
## go up and down the dispersion and find the aperture center at
## each position.  'ridgeline' does this -- 'columns' are the centers
## of the windows, 'profctrs' the centers of the aperture found at
## these positions, i.e., the points found in IRAF's 'aptrace'.

    columns, profctrs = ridgeline(data, config['DISPLINE'],
       config['NAVG'], apcenter, apoffset, aplow, aphigh, 
       nord, niterate, clip) # ,verbose=True)  

## Plot out and fit a polynomial to the aperture trace.

    fig2 = plt.figure(figsize=(config['FIGWID'],config['FIGHGHT']))
    cid = fig2.canvas.mpl_connect('key_press_event',onpress2)
    fig2.set_size_inches(config['FIGWID'],config['FIGHGHT'])

    origcolumns = columns
    origprofctrs = profctrs
    
    # line instances for the column fit
    colpts = None 
    colfitline  = None 

    plt.plot(columns,profctrs,'bo') 
    plt.xlabel("Pixel along dispersion")
    plt.ylabel("Aperture centroid")

    # the fitting, iterations, etc are all done in this 
    # routine

    scaledcoeffs = fittrace(traceord, traceiter) 

    # and the adjustments in the plt.show() using cursor
    # commands.  What we need for output is 'scaledcoeffs',
    # which is defined as global in 'onpress2' and is therefore
    # adjusted in the global program.

    if not args.defaultcursor :
       axes = plt.gca()
       cursor = Cursor(axes,useblit=True,color='red',linewidth=1)
    plt.show()

    # Write out the results!

    write_iraf_legendre(im, apcenter, aplow, aphigh, 
         b1rel, b2rel, b3rel, b4rel, 
         scaledcoeffs, path = './database') 

    hdu.close()

