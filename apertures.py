#!/usr/bin/env python3
"""apertures.py -- Python code to replace apedit and aptrace.  
     Starting with proof-of-concept.
"""

import astropy.io.fits as fits
import matplotlib as mpl
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
#  DESCRIPTION AND SHORT MANUAL
# 
#  Originally by John Thorstensen, Dartmouth College;
#  first version is from 2021 August.
#
# apertures.py -- Standalone python 3 program to replace
#   the iraf 'apedit' and 'aptrace' tasks, for spectra 
#   of single stellar targets.
# 
#   Like 'apedit' and 'aptrace', this code is used to 
#   define software apertures for extracting the 1-d target 
#   spectra from the 2-d images. Again like 'apedit' and 
#   'aptrace', theis program does not actually perform the
#   extraction.  Instead, it writes its results in a text
#   file that can be interpreted by extraction programs.
#   After defining the software apertures, you can extract
#   the spectra using the iraf 'apsum' task, or, if it is
#   suitable for your data, my own program 'opextract.py'.
#
#   Finding and tracing software apertures is not difficult
#   for isolated, bright targets, and non-interactive batch
#   processing should work fine.  However, crowding and/or
#   poor signal-to-noise makes a more challenging task, and
#   human intervention can be important. 
#   This particular program forces you to look at all the 
#   data you're going to extract, and makes it possible to 
#   edit the apertures with cursor-driven commands, 
#   reminiscent of 'apedit' and 'aptrace' in interactive mode..  
#
#   Limitations:
#
#      - Assumes there is only one target.
#      - Assumes the target is reliably centered somewhere
#          in a modest range of pixels along the slit.
#      - While not a formal requirement, it'll probably work 
#          best with point sources.
#      - Does all curve fits with legendre polynomials; 
#          cubic splines and chebyshev polynomials are not
#          implemented.
#
#   How to use it:
#
#      - Copy 'apertures.config' into the directories 
#          where your images reside.  This has adjustable  
#          parameters similar to the parameter files in 
#          iraf/pyraf.  The format is simple, namely
#
#            parameter | value | datatype
# 
#          In what follows, parameters from this file are 
#          listed in UPPERCASE.
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
#          Note especially the '-c' command line argument; use
#          this if the dispersion runs along columns rather 
#          that along rows (so spectra are vertical stripes).
#      -  The program will compute an average profile around
#          DISPLINE, which you can choose to be the pixel value of 
#          at a wavelength at which e you expect your sources to 
#          be easy to find.  The number of individual lines 
#          averaged is NAVG.  The average can either be a median,
#          or an average with low and high pixels rejected.
#      -  A subarray of width PROFWIN centered on PROFCTR is extracted 
#          from the whole line profile.  It is searched
#          for the maximum value, which is a tentative center.
#      -  The background is computed within limits B1 and B2 on the 
#          left, and B3 and B4 on the right; these values are all 
#          in units of pixels relative to the peak (so B1 and B2 are
#          negative numbers).  The fit is iterated several times
#          which generally removes other stars in the slit, glitches, etc.
#          The background fit parameters can be changed in the config
#          file but not interactively.  The limits of the background
#          region can be changed using the cursor (more later). 
#      -  The fitted background is subtracted from the profile subarray.
#      -  The aperture limits are found by 'walking' down from the 
#           peak of the background-subtracted subarray on either side
#           of the peak until a value is reached that is less than YLEVEL 
#           times the peak.  The limit on either side is then computed 
#           by 'growing' the width on either side by a factor of R_GROW.  
#           This is very similar to the iraf algorithm.
#      -  The center of the profile is refined by finding the first 
#           first moment of the slit profile within the aperture limits.
#      -  A plot is presented showing the average counts with position along
#           the slit.  Also shown are:
#           - the computed profile center marked by a solid line
#           - the lower and upper aperture limits (dashed lines)
#           - the limits of the subarray used for profile finding etc,
#               (dotted lines)
#           - the fit to the background
#           - the points used for the background fit (not rejected by
#             iterations) are overplotted in red.
#           - unless you turn it off at the command line, a 'fancy cursor'
#               appears.
#      -  You can now make changes using the following cursor commands;
#           to make an adjustment, set the cursor to the desired spot and type
#           the appropriate character.  (This is just like iraf).
#           '?' will print a menu of the commands:
#           The graph is updated and redrawn immediately after each keystroke.
#             - 1, 2, 3, and 4 -- set background limits and recompute.
#             - c -- recenter the aperture on the cursor.
#             - l and u -- change the lower and upper aperture limit.
#           You can change the plot limits : 
#             - w pans out to whole profile.
#             - r restores default plot limits.
#             - j, k - set left and right plot limits
#             - t, b - set top and bottom plot limits.
#           Finally : q - quit and accepts the settings you've done.
#      
#      After the aperture and background regions are defined, the program 
#         executes something closely resembling iraf's
#        'aptrace' -- it runs up and down the dispersion and finds the 
#         aperture center at each wavelength.  It then fits a polynomial
#         of order TRACEORD to the points, and iterates TRACEITER times,
#         eliminating points outside of TRACECLIP sigma.  
#
#      The trace is presented in a plot, with (again) the points used
#         overplotted in red.  One can adjust the fit:
#             - d  Delete the point nearest the cursor and refit.
#             - r  Restore the point nearest the cursor and refit.
#             - R  Restore ALL the deleted points, and refit.
#             - )  Delete ALL points to the RIGHT of the cursor and refit.
#                   This is useful if the spectrum fades off at the end,
#                   which manifests as a 'fraying' of the points.
#             - (  Delete ALL points to the LEFT  of the cursor and refit.
#             - o or O (lower and upper case) respectively decrease or
#                 increase the order of the fitted polynomial by 1.
#             - i  iterate the fit the default number of times and 
#                   reject discrepant pixels at each iteration.
#             - ?  Print a menu.
#             - q  Quit the fitting and proceed.
#        As with the profile fitting, the graph replotted after each command,
#          showing any changes..
#        NOTE that with d, r, R, ), and (, no further points are rejected.
#              but with o, O, and i, the iteration/rejection cycle is run.
#
#      When you exit with 'q', the program writes out the aperture
#        information to a file named 'ap' + (image root),
#        e.g. apccd123 for image 'ccd123', by default in the 'database' 
#        subdirectory under your current directory; if the 'database' 
#        subdirectory doesn't exist, it is created.  Any previous 'ap' 
#        file is overwritten.
#  
#      The format of the database 'ap' file is designed to be compatible with
#        the format used by iraf.  This is understood by my 'opextract.py'
#        program.  Either iraf's 'apsum' or my own program can now be used to
#        extract the spectrum.   
#
#    If you're doing an '@' list, the program moves on to the next object. 
#
############################################################################

def legoverdomain(xarr, yarr, domainends, order) : 

    # numpy's 'legfit' just fits the values.  We want to write
    # a output that's compatible with input to IRAF, which 
    # wants (a) the ends of the domain and (b) legendre coeffs 
    # that are appropriate for x-values rescaled to [-1,1]

    # This routine takes x and y point arrays, and a tuple 
    # giving (xmin,xmax), rescales the x-values, and then 
    # does the fit.  I'm passing in the domain limits separately 
    # to preserve the full range of x; xarr has often been 
    # truncated by iterations.
    
    # the 'order' is the standard order (e.g. order 2 = parabolic).
    # IRAF uses an 'order' that is larger by 1, so a parabola
    # would have iraforder 3.

    # Recenter the array on the mean of domainends,
    xarr = xarr - (domainends[1] + domainends[0]) / 2.
    # and rescale.
    xarr = xarr * 2. / (domainends[1] - domainends[0]) 
    
    # fit the scaled array.

    scaledcoeffs = leg.legfit(xarr,yarr,order) 

    return scaledcoeffs

def evaluate_legdomain(xarr, scaledcoeffs, domainends) :

    # numpy's Legendre object can be set up with scaled coeffs 
    # and a domain, so this is easy.

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

    # This assumes only one target per image.  

    # print(f"path is {path}")

    if not os.path.exists(path) :
        try :
             os.makedirs(path)
        except :
             print(f"Could not make the directory {path}!")

    if '.fits' in imname :
        imname = imname.replace('.fits','')

    # assumes unix/linux path separator.  So sue me.
    if path[-1] == '/' :
        outfname = path + 'ap' + imname
    else : 
        outfname = path + '/ap' + imname
    
    # record the time and date.

    timenow = datetime.datetime.now()
    # using y-m-d, inverting ancient IRAF format.
    timestring = timenow.strftime("%a %H:%M:%S %Y-%b-%d")
    outf = open(outfname,"w")
    outf.write(f"# {timestring}\n")

    dl = config['DISPLINE']  # for brevity

    # if dispersion is along columns, the numbers change oddly.
    if args.columndispers : 
        outf.write(f"begin\taperture {imname} 1 {apcenter:7.3f} {data.shape[1] - dl}\n")
    else :
        outf.write(f"begin\taperture {imname} 1 {dl:5.0f} {apcenter:7.3f}\n")

    # The leading '\t' has to be stripped out in all these because it somehow 
    # breaks the f-string and throws an error.

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
    outf.write("\n")   # seems to like a blank line at the end.

    outf.close()

def getconfig(infname) :
    # read in the parameters from the config file.

    # Format is  PARAM | value | datatype

    try : 
        inf = open(infname,'r')
    except : 
        print(f"Could not open config file {infname} - dying.")
        # this really is fatal.  Die!
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

    # Interprets the positional argument.  To pass in a list of 
    # images, # prepend the name with "@"; otherwise it will look for a
    # single file of the specified name.  Filenames can be given with or
    # without the '.fits' suffix; however, the actual files need to 
    # have the '.fits' suffix, since if the name does not have 
    # '.fits', it will be appended to the name.

    frootlist = []    # filename roots
    imnamelist = []   # actual image names with '.fits'

    if instring[0] == '@' :   # It's a list
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

    else :  # "I don't like bombastic late Romantic piano music, 
            # said Tom Lisztlessly." (It's a simple filename)

        if '.fits' in instring :
            imnamelist.append(instring)
            frootlist.append(instring.replace('.fits',''))
        else :
            imnamelist.append(instring + '.fits')
            frootlist.append(instring)

    return(frootlist, imnamelist) 

def clippedavg(arr, n_low_cut, n_high_cut) :

    # Averages an array after rejecting the largest n_low_cut
    #  points and the highest n_high_cut points.

    if(n_low_cut + n_high_cut >= len(arr)) :
        print("Number of points to cut bigger than array size!")
        return np.average(arr, axis = 1)
    if(n_low_cut < 0 or n_high_cut < 0) :
        print("Number of points to cut must be positive or zero.")
        return np.average(arr, axis = 1)
    
    sorted_copy = np.sort(arr) 

    # use index range (along axis 1, not axis 0) in the sorted array to 
    # exclude the high 
    
    return np.average(sorted_copy[: , n_low_cut : -1 * n_high_cut], axis = 1)

def avgcols(data, midcol, navg, avgtype = "median", n_low_cut=0, n_high_cut=0) :

    # returns an array of 1-indexed row numbers and the
    # average over the specified columns.
    # [dispersion is along rows.  If the input data has 
    # dispersion along columns it should already have been
    # transposed, i.e., flipped over.]

    # 'avgtype' can be "median" or "clippedavg".  If it is clippedavg,
    # the average is computed after the smallest n_low_cut pixel values
    # and the highest n_high_cut pixel values are rejected.  Obviously,
    # n_low_cut and n_high_cut should be fairly small compared to navg.
    # There's no noticeable difference in speed between the algorithms.

    # make a 1-INDEXED array of pixel numbers.
    xdisp = np.arange(1,data.shape[0]+1) 

    # make and check limits for averaging

    if midcol < data.shape[1] and midcol > 0 : 
        startcol = int(midcol - navg / 2)
        if startcol < 0 : startcol = 0
        endcol = int(midcol + navg / 2)
        if endcol > data.shape[1] : endcol = data.shape[1] - 1

        if "median" in avgtype : 
            return (xdisp,np.median(data[0:data.shape[0], startcol:endcol,],axis = 1))

        elif "clippedavg" in avgtype :
            return (xdisp,clippedavg(data[0:data.shape[0], startcol:endcol,],
                   n_low_cut = 2, n_high_cut = 2))
    else:
        print("column to average is not within bounds.")
        return (xdisp,np.zeros(data.shape[0]))

def fitsky(xrows, profile, b1, b2, b3, b4, nord, niterate, clip, 
    doplot=True, verbose = True) :

    # fit the sky background along a dispersion line, in 
    # windows on either side, with iteration to reject outliers.

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

        if stdev < 0.1 :  # there's something wrong.
            print("Breaking - stdev < 0.1")
            break;

        # numpy's fancy indexing
        #  makes it easy to remove the outliers.

        keepindices = abs(residuals) <= clip * stdev
        xarr = xarr[keepindices]
        yarr = yarr[keepindices]

        if verbose : print(f"bckg iteration {niter}; {len(xarr)} points.")

    # keep the value of the background at aperture center.
    bckgatctr = leg.legval(int(indmax),legcoefs)
    if verbose : print(f"bckgatctr {bckgatctr}")

    # evaluate the fit over all the rows and optionally plot.

    bckgfit = leg.legval(xrows,legcoefs)

    if doplot : 
        # This is tricky -- 'bckgfitline' is the artist or
        # whatever that drew the line.  First need to remove
        # the old line if it exists:
        if bckgfitline is not None : 
            bfitl = bckgfitline.pop(0) # it's a tuple
            bfitl.remove()
        # and plot the new one.
        bckgfitline = plt.plot(xrows, bckgfit) 
   
        # and same for the points that survived iteration:

        if bckgpts is not None : 
            bpts = bckgpts.pop(0) # it's a tuple
            bpts.remove()
        bckgpts = plt.plot(xarr, yarr,'ro',markersize=3)

    return (bckgfit, bckgatctr) 

def fittrace(traceord, traceiter, doplot = True,
    verbose = False) :

    # Fits a legendre to the aperture center as a function
    # of position along the dispersion, and optionally 
    # plots.

    global colpts, colfitline
    global columns, profctrs

    # do a single fit outside the iteration loop if traceiter is 0.

    if traceiter == 0 :
        legcoefs = leg.legfit(columns, profctrs, traceord)
        fit = leg.legval(columns,legcoefs)
        residuals = profctrs - fit
        stdev = np.std(residuals) 
    
    else :

        # the iteration scheme is similar to that used for 
        # the background points.  See 'fitsky' for far too much
        # commentary.

        for i in range(0,traceiter) :
    
            legcoefs = leg.legfit(columns, profctrs, traceord)
            fit = leg.legval(columns,legcoefs)
            residuals = profctrs - fit
            stdev = np.std(residuals) 
    
            if stdev < 0.001 :
                print("Breaking -  stdev < 0.001")
                break
            else :
                keepindices = abs(residuals) <= traceclip * stdev
                columns = columns[keepindices]
                profctrs = profctrs[keepindices]
                if verbose : 
                    print(f"trace iteration {i}; {len(columns)} points.")

    # Once you have the points to fit and so on, recast as an IRAF-style
    # scaled legendre.
        
    scaledcoeffs =  legoverdomain(columns, profctrs, (1, data.shape[1] - 1), traceord)  
    fit = evaluate_legdomain(origcolumns, scaledcoeffs, (1, data.shape[1] - 1))

    if doplot : 

        if colfitline is not None :  # delete old artists, as with profile 
            colfitl = colfitline.pop(0) # it's a tuple
            colfitl.remove()
        if colpts is not None :
            colptsl = colpts.pop(0)
            colptsl.remove()

        # plot the fit over the whole range of columns.
        colfitline = plt.plot(origcolumns,fit) 
        # and overplot in red the points actually used in the fit.
        colpts = plt.plot(columns, profctrs, 'ro',markersize=3)

        plt.title(f"{im} {hdr['object']} - keeping {len(columns)}. order {traceord}, iterated {traceiter}, stdev {stdev: 5.2f}")
        
    return scaledcoeffs
 
def onpress(event) :  # This does all the human interaction for the
    # aperture plot.

    # these are artists that may need to be deleted or revised
    # in replotting.

    global bckgpts
    global bckgfitline
    global centerline, lowapline, highapline
    
    global xlow, xhigh, ylow, yhigh   # actual plot limits.
  
    # These are the results of the interactive cursor stuff;
    # I can't figure out how to hand them back to the main 
    # program except by declaring them as global.

    global b1, b2, b3, b4
    global apcenter, aplow, aphigh

    # Use 1, 2, 3, or 4 to change background limits and replot.

    # note that the changes will not be displayed without a 'plt.draw()'.

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

    if event.key == 'c' :  # set aperture center to the cursor location
        apcenter = event.xdata
        centerline.remove()
        centerline = plt.axvline(apcenter)
        plt.draw()

    # 'l' and 'u' set upper and lower liits of aperture.  This required
    # deleting the default matplotlib key binding of 'l', which otherwise
    # sets the y scale to logarithmic.

    if event.key == 'l' :
        aplow = event.xdata
        lowapline.remove()
        lowapline = plt.axvline(aplow, color='green', linestyle = 'dashed')
        plt.draw()
    if event.key == 'u' :
        aphigh = event.xdata
        highapline.remove()
        highapline = plt.axvline(aphigh, color='green',linestyle = 'dashed')
        plt.draw()

    # commands to change limits
    if event.key == 'w' :  # look at whole range 
        axes = plt.gca()
        axes.set_xlim((xlow_wide,xhigh_wide))
        axes.set_ylim((ylow_wide,yhigh_wide))
        plt.draw()

    if event.key == 'r' :  # restore original limits
        axes = plt.gca()
        axes.set_xlim((xlow_default,xhigh_default))
        axes.set_ylim((ylow_default,yhigh_default))
        plt.draw()

    if event.key == 'j' : # set left plot limit
        axes = plt.gca()
        xlow = event.xdata
        axes.set_xlim((xlow,xhigh))
        plt.draw()

    if event.key == 'k' : # set right plot limit
        axes = plt.gca()
        xhigh = event.xdata
        axes.set_xlim((xlow,xhigh))
        plt.draw()
 
    if event.key == 'b' : # set bottom plot limit
        axes = plt.gca()
        ylow = event.ydata
        axes.set_ylim((ylow,yhigh))
        plt.draw()

    if event.key == 't' : # set top plot limit
        axes = plt.gca()
        yhigh = event.ydata
        axes.set_ylim((ylow,yhigh))
        plt.draw()
            
    # print out a summary if needed.

    if event.key == 'p' :
        print(f"[low : center : high] = [{aplow:5.1f} : {apcenter:5.1f} : {aphigh:5.1f}]")
        print(f"background: [{b1:4.0f}:{b2:4.0f},{b3:4.0f}:{b4:4.0f}]")

    if event.key == '?' :
        print("Aperture finding key commands:")
        print("c - Center aperture on cursor.")
        print("l - Set lower aperture limit.")
        print("u - Set upper aperture limit.")
        print("1 - set lower left background limit")  
        print("2 - set lower right background limit")  
        print("3 - set upper left background limit")  
        print("4 - set upper right background limit")  
        print("p - print current paramters.")
        print("Commands to set plot limits:")
        print("w - pan out to whole profile.")
        print("r - restore default plot limits.")
        print("j, k - set left, right plot limits.")
        print("t, b - set top, bottom plot limits.")
        print("q - accept fit and quit.")

    # Typing a 'q' exits the event loop; I don't have a provision for aborting
    # without saving.

  #  if event.key == 'q' :
  #      print('got q.')

def findnearest(x, y, xpts, ypts, xspan, yspan) : 

    # picks the point among (xpts, ypts) nearest to the given 
    # x and y (from the cursor) on a graph with ranges
    # xspan and yspan.  I know point-picking is available with 
    # the mouse but I will also need access to the character typed.

    # spans are needed because the scales of the axes are
    # often wildly different, so one wants to convert to something 
    # like the distance on the screen.  The graph is generally
    # not square, but this is close enough.

    # xpts and ypts are numpy arrays, so can vectorize.

    dxnorm = (xpts - x) / xspan
    dynorm = (ypts - y) / yspan
  
    # reference for this next step is Pythagoras (-500); 
    # no need to take sqrt since we just want the minimum index.

    misses = dxnorm * dxnorm + dynorm * dynorm  
  
    return np.argmin(misses)     

def onpress2(event) :  # This does the human interaction for the trace.

    # results are in global variables, because I can't figure
    # out any other way of getting this info back to the main 
    # program since it's generated in an event loop.

    global columns, profctrs, traceord
    global scaledcoeffs

    # 'lines' that may need to be deleted to revise the plot
    global colpts
    global tracefitline

    # note that 'O' and 'o' refit and re-iterate; the iteraction
    # cuts points but never adds them.
    
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
        axes = plt.gca()
        xlims = axes.get_xlim()
        ylims = axes.get_ylim()
        killindex = findnearest(event.xdata, event.ydata, columns, profctrs, 
           xlims[1] - xlims[0], ylims[1] - ylims[0])
        columns = np.delete(columns, killindex)
        profctrs = np.delete(profctrs, killindex)
        
        # refit without iteration.
        scaledcoeffs = fittrace(traceord,0) 
        plt.draw()

    if event.key == 'r' :

        # restore point nearest to cursor.
        axes = plt.gca()
        xlims = axes.get_xlim()
        ylims = axes.get_ylim()
        addindex = findnearest(event.xdata, event.ydata, origcolumns, origprofctrs, 
           xlims[1] - xlims[0], ylims[1] - ylims[0])

        # appending these means they won't be in order any more.  This doesn't seem
        # to cause any issues.
        columns = np.append(columns, origcolumns[addindex]) 
        profctrs = np.append(profctrs, origprofctrs[addindex])

        # refit without iteration, or else the point gets thrown out again!
        scaledcoeffs = fittrace(traceord,0) 
        plt.draw()

    if event.key == 'R' :

        # restore ALL points, fit without iteration.
        # This is needed in case previous fits have eliminated too many points

        columns = origcolumns 
        profctrs = origprofctrs

        # refit WITHOUT iterating, or else points get thrown out again!
        scaledcoeffs = fittrace(traceord,0) 
        plt.draw()

    if event.key == 'i' :

        # just fit, WITH iteration.  This would usually be done after an 'R'
        # that restores all the points.

        scaledcoeffs = fittrace(traceord,traceiter) 
        plt.draw()

    # The profile is sometimes lost beyond a certain point; allow
    # wholesale deletion of points in this case.

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
        # collect indices of points to kill, and then delete them all
        # at once; if you try to do them one-by-one it changes the 
        # indexing as you and you can run off the end of the array.

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
  nord, niterate, clip,     # parameters for background fitting
  avgtype = "median", n_low_cut = 0, n_high_cut = 0,  # parameters for
                            # profile averaging
  verbose = False) : 

# walks up and down the dispersion and finds profile at each location
# non-interactively, i.e., the points to be fitted for the trace.
     
    colnum = startline
    lastctr = startcenter

    # Figure out relative aperture limits so that search window can
    # move with the spectrum.

    relaplow = aplow - lastctr 
    relaphigh = aphigh - startcenter 
    if verbose : print(f"relaplow high {relaplow} {relaphigh}") 

    ctrdict = {}  # indexed by column; to facilitate sorting at end.
    ctrdict[startline] = startcenter

    # step through columns, compute profile centers.

    while colnum < data.shape[1] :
       # colnum = colnum + colwidth  # Don't update colnum until the end -- this avoids
       # the need for the awkward 'break' statement and ensures that the computation is 
       # exactly the same for the displine as for all the others.
       # if colnum > data.shape[1] : break
       xrows, profile = avgcols(data, colnum, colwidth,
                 avgtype=avgtype, n_low_cut = n_low_cut, n_high_cut = n_high_cut) 
       if verbose : print(f"colnum {colnum} lastctr {lastctr} aplow {aplow} aphigh {aphigh}")
       aplow = lastctr + relaplow
       aphigh = lastctr + relaphigh

       # The aperture center is computed as a first-moment.  This has the consequence that
       # at high S/N, if the aperture limits don't include every bit of flux, one will see slight
       # discontinuities in the aperture center when a new pixel is included as we step through
       # columns.  This should should have no appreciable effect.  If it is a problem, then 
       # either (a) write a more elaborate scheme that deals with fractional pixels at the edge,
       # or (b) find another centering algorithm.  (I use a convolution method for
       # line centering that might be repurposed; look for Schneider and Young 1980.)
 
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
       
       # somehow managed not to repeat the squaring bug here.
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

       # NOTE that aperture limits are not recomputed -- width is constant along 
       #  the dispersion.  This may be an issue of the optics of the spectrograph
       #  are wonky. It's also thought that seeing does vary with wavelength somewhat.

    # step down through columns from start.  Lots of copypasta
    # but I'm too lazy to write a function.  Reset back to beginning:
              
    colnum = startline - colwidth        
    lastctr = startcenter

    while colnum > 0 :
       
       colnum = colnum - colwidth  
       if colnum < 0 : break

       xrows, profile = avgcols(data, colnum, colwidth,
                 avgtype=avgtype, n_low_cut = n_low_cut, n_high_cut = n_high_cut) 

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
    # There's probably some fabulous pythonic way to do these next two steps.
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

# And away we go!

for im in imlist :

    hdu = fits.open(im)
    hdr = hdu[0].header
    data = hdu[0].data

    fig = plt.figure(figsize=(config['FIGWID'],config['FIGHGHT']))
    cid = fig.canvas.mpl_connect('key_press_event',onpress)

    # On the first pass, disable some matplotlib key bindings we'd like to 
    # use for cursor commands -- see 
    # https://stackoverflow.com/questions/35624183/disable-matplotlibs-default-arrow-key-bindings
    # This is wrapped in a try/except because when you do the second image in a list,
    # the keymaps have already been removed, causing a crash.
  
    try :
        mpl.rcParams['keymap.all_axes'].remove('a')
        mpl.rcParams['keymap.xscale'].remove('k')
        mpl.rcParams['keymap.yscale'].remove('l')
        mpl.rcParams['keymap.zoom'].remove('o')
        print("Disabled selected keymaps.")
    except : 
        pass
    
    # Transpose the data if dispersion runs along columns.
    # This transposition, and some rearrangement of the numbers in the output file,
    # is apparently all that's needed to cover this case.

    if args.columndispers : 
        data = data.T    

    # Take the median NAVG dispersion lines near DISPLINE.
    # Note that xrows is a 1-INDEXED array of pixel locations
    # along the slit, so xrow[0] = 1., xrow[1] = 2., and so on.
    # IRAF is one-indexed and the result of this is that the positions
    # returned by the program are not off-by-one from IRAF.

    avgtype = config['AVGTYPE']        # how to average dispersion lines
                                       # either 'median' or 'clippedavg'.
    n_low_cut = config['NLOWCUT']      # number of low pixels to cut in clippedavg
    n_high_cut = config['NHIGHCUT']    # same for high pixels          

    xrows, profile = avgcols(data,config['DISPLINE'],config['NAVG'],
         avgtype = avgtype, n_low_cut = n_low_cut, n_high_cut = n_high_cut)

    # Search for the profile ONLY in a window around where the target is 
    # expected. In my own data the intended target is always close to the 
    # same location  on the slit.  
    # IRAF 'apfind' always finds the brightest object, which is 
    # often the wrong star, requiring deletion, re-marking, etc.
    # This simple change should avoid that.

    # If your target location is inconsistent, simply increase PROFWIN
    # in the .config file.

    searchstart = int(config['PROFCTR'] - config['PROFWIN'] / 2.)
    searchend =   int(config['PROFCTR'] + config['PROFWIN'] / 2.)

    # first estimate of center is location of max flux in the window.

    fmax = -10000000.
    indmax = 0
    for i in range(searchstart,searchend) : 
        if profile[i] > fmax : 
           fmax = profile[i]
           indmax = xrows[i]  # not 'i' itself -- else off by one!

    plt.plot(xrows,profile)
    plt.xlabel("Pixel across dispersion")
    plt.ylabel("Counts")
    
    # get and store x and y limits for later rescaling if requested
    axes = plt.gca()
    (xlow_wide, xhigh_wide) = axes.get_xlim()
    (ylow_wide, yhigh_wide) = axes.get_ylim()

    # set plot limits to include the background regions on either side,
    # and a little more.

    bckrange = config['B4'] - config['B1']  # full width of background regions
    xlow = indmax + config['B1'] - 0.05 * bckrange
    xhigh = indmax + config['B4'] + 0.05 * bckrange

    # store limits for later rescaling if requested
    xlow_default = xlow
    xhigh_default = xhigh 

    # fit the background. Note B1, B2 etc are relative to aperture,
    # while b1, b2 etc are relative to whole array.

    b1 = int(indmax + config['B1'])
    b2 = int(indmax + config['B2'])
    b3 = int(indmax + config['B3'])
    b4 = int(indmax + config['B4'])

    niterate = config['BITER']
    nord = config['BFITORD']
    clip = config['BSIG']

    # initialize these 'artists' or whatever -- they get stored when 
    # the plot is drawn, and then erased if the user changes the limits etc.
 
    bckgpts = None      # line instance of dots showing points used for bckgrd.
    bckgfitline = None  # line instance for plotted background fit.

    (bckgfit, bckgatctr) = fitsky(xrows, profile, b1, b2, b3, b4, nord,
        niterate, clip, verbose = False) 

    # scale y axis
    yrange = fmax - bckgatctr
    ylow = bckgatctr - 0.15 * yrange
    yhigh = fmax + 0.05 * yrange

    ylow_default = ylow
    yhigh_default = yhigh

    # adjust the region of interest to center on the actual max
    # found in the profile, in case the star was not placed
    # exactly in the fiducial spot on the slit.

    lowapind = int(indmax - config['WIDTH'] / 2)
    highapind = int(indmax + config['WIDTH'] / 2)

    # Find the aperture center as the FIRST MOMENT of the 
    # profile within the region of interest.

    # apy will be a subarray containing only the region around 
    # the profile.

    apy = profile[lowapind:highapind] - bckgfit[lowapind:highapind]

#     wgt = apy * apy  # oops -- this was apparently a bug!

    wgt = apy  # You don't square the y-value for a moment calculation.
               # the new variable is unneeded but harmless.

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

    # try the iraf method - walk down from max to YLEVEL which is 
    # a specified fraction of the peak value;  
    # then 'grow' the limits by the factor R_GROW.

    fidlevel = config['YLEVEL']  # something like 0.3 typically
    apgrow = config['R_GROW']    # maybe 1.5 

    apymaxind = np.argmax(apy)
    maxval = apy[apymaxind]
    crity = fidlevel * maxval  # level at desired 'fidlevel'

    # walk out from center index until less than crity:  

    # left side
    j = apymaxind
    while apy[j] > crity and j > 0 : j = j - 1
    # this might be refined with an interpolation but it
    # seems good enough.
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

    # Note that there's no attempt at sub-pixel interpolation here.

    apy = profile[int(aplow):int(aphigh)] - bckgfit[int(aplow):int(aphigh)]
    # wgt = apy * apy   # that bug again (which was nearly harmless in practice)
    wgt = apy    
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
       nord, niterate, clip,
       avgtype = avgtype, n_low_cut = n_low_cut, n_high_cut = n_high_cut) 
       # ,verbose=True)  

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
    # routine; pull in default parameters first.

    traceord = config['TRACEORD']
    traceiter = config['TRACEITER'] 
    traceclip = config['TRACECLIP']

    scaledcoeffs = fittrace(traceord, traceiter) 

    # All the hand-adjustments to this are done in the 
    #the plt.show() using cursor commands.  

    # The only output is 'scaledcoeffs', which is defined 
    # as global in 'onpress2' and is therefore
    # adjusted in the global program.  This is the only
    # way I could figure out to pass back information from
    # the graphics event loop.

    # use the fancy cursor by default:
    if not args.defaultcursor :
       axes = plt.gca()
       cursor = Cursor(axes,useblit=True,color='red',linewidth=1)

    # Aaand, as with the profile paramter graph, the plt.show()
    # enters an event loop in which you can fuss with the
    # fit.

    plt.show()

    # As soon as you quit the graph, you're ready to write!

    write_iraf_legendre(im, apcenter, aplow, aphigh, 
         b1rel, b2rel, b3rel, b4rel, 
         scaledcoeffs, path = './database') 

    # close up and do the next image.

    hdu.close()

