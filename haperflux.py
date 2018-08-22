

"""
=====================================================
haperflux.py : Circular aperture photmetry
=====================================================

This module provides circular aperture photometry
functionality for Healpix maps. It is a Python
implementation of the original IDL code written
by Clive Dickinson, et al, available at:

http://irsa.ipac.caltech.edu/data/Planck/release_1/software/

- :func:'convertToJy' convert units to Janskys
- :func:'planckcorr' conversion factor between CMB_K and Jy
- :func:'haperflux' gets flux of a single aperture
- :func:'haperfluxMany' gets fluxes over multiple regions and maps

"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import astropy.io.fits as fits
import sys
import math
from astropy.stats import mad_std
from astropy import units as u
from astropy.coordinates import SkyCoord

def planckcorr(freq):
    h = 6.62606957E-34
    k = 1.3806488E-23
    T = 2.725
    x = h*(freq*1e9)/k/T
    return (np.exp(x) - 1)**2/x**2/np.exp(x)

def convertToJy(units, thisfreq, npix):

    # get conversion factors for an aperture to Jy/pix, from any of the following units:
    # Kelvin(RJ), Kelvin(CMB), MJy/sr, Average
    pix_area = 4.*np.pi / npix

    factor = 1.0

    if (units == 'K') or (units == 'K_RJ') or (units == 'KRJ'):
        factor = 2.*1381.*(thisfreq*1.0e9)**2/(2.997e8)**2 * pix_area

    elif (units == 'mK') or (units == 'mK_RJ') or (units == 'mKRJ'):
        factor = 2.*1381.*(thisfreq*1.0e9)**2/(2.997e8)**2 * pix_area / 1.0e3

    elif (units == 'uK') or (units == 'uK_RJ') or (units == 'uKRJ'):
        factor = 2.*1381.*(thisfreq*1.0e9)**2/(2.997e8)**2 * pix_area / 1.0e6

    elif (units == 'K_CMB') or (units == 'KCMB'):
        factor = 2.*1381.*(thisfreq*1.0e9)**2/(2.997e8)**2 * pix_area / planckcorr(thisfreq)

    elif (units == 'mK_CMB') or (units == 'mKCMB'):
        factor = 2.*1381.*(thisfreq*1.0e9)**2/(2.997e8)**2 * pix_area / 1.0e3 / planckcorr(thisfreq)

    elif (units == 'uK_CMB') or (units == 'uKCMB'):
        factor = 2.*1381.*(thisfreq*1.0e9)**2/(2.997e8)**2 * pix_area / 1.0e6 / planckcorr(thisfreq)

    elif (units == 'MJy/sr') or (units == 'MJY/SR') or (units == 'MjySr'):
        factor = pix_area * 1.0e6

    elif (units == 'Jy/pixel') or (units == 'JY/PIXEL') or (units == 'JY/PIX') or (units == 'JyPix'):
        factor = 1.0

    elif (units == 'average') or (units == 'avg') or (units == 'Average') or (units == 'AVG'):
        factor = 1.0 / float(ninnerpix)
    else:
        print "Invalid units specified! "

    return factor

def haperflux(inmap, freq, lon, lat, aper_inner_radius, aper_outer_radius1, aper_outer_radius2, \
              units, fd=0, fd_err=0, fd_bg=0, column=0, dopol=False, nested=False, noise_model=0, centroid=False, arcmin=True):

    """ Do aperture photometry on a Healpix map with a circle of a given radius, and subtracting the background in an annulus
    Units are converted to Jy.

    Parameters
    -----------
    inmap : Healpix fits file or map array
    freq : Frequency (GHz) to allow conversion to Jy
    lon : Longitude of aperture (deg)
    lat : Latitude of aperture (deg)
    aper_inner_radius :   Inner radius of aperture (arcmin)
    aper_outer_radius1 : 1st Outer radius of aperture beteween aperture
        and  b/g annulus (arcmin) Make this the same as the inner radius for
        single annulus with no gap.
    aper_outer_radius2 : 2nd Outer radius of aperture for b/g annulus (arcmin)
    units : String defining units in the map.
        Options include ['K','K_RJ', 'K_CMB', 'MJy/sr','Jy/pixel']
        m for milli and u for micro can also be used for
        K e.g. 'mK_RJ' or 'mK'. Default is 'K_RJ'
    nested : If set, then the ordering is NESTED
        (default is to assume RING if not reading in a file)
    noise_model : Noise model for estimating the uncertainty
        0 (DEFAULT) = approx uncertainty for typical/bg annulus aperture
        sizes (only approximate!).
        1 = assumes white uncorrelated noise (exact)
        and will under-estimate in most cases with real backgrounds!

    Returns
    --------
    fd : Source flux density [Jy] (Background-subtracted)
    fd_err : Noise level (det. from backgorund ring)
    fd_bg : Background level
    """

    #set parameters
    inmap = inmap
    thisfreq = float(freq) # [GHz]
    lon = float(lon) # [deg]
    lat = float(lat) # [deg]
    aper_inner_radius  = float(aper_inner_radius)  # [arcmin]
    aper_outer_radius1 = float(aper_outer_radius1) # [arcmin]
    aper_outer_radius2 = float(aper_outer_radius2) # [arcmin]

    # Convert the apertures to radians, if given in arcminutes.
    #     The pixel-finder 'query_disc' expects radians:
    if arcmin == True:

        aper_inner_radius  = aper_inner_radius/60.*np.pi/180.  # [rad]
        aper_outer_radius1 = aper_outer_radius1/60.*np.pi/180. # [rad]
        aper_outer_radius2 = aper_outer_radius2/60.*np.pi/180. # [rad]

    # read in data
    s = np.size(inmap)

    if (s == 1):
        print "Filename given as input..."
        print "Reading HEALPix fits file into a numpy array"

        hmap,hhead = hp.read_map(inmap, hdu=1,h=True, nest=nested)

    if (s>1):
        hmap = inmap

    if (nested==False):
        ordering='RING'
    else:
        ordering ='NESTED'

    nside = np.sqrt(len(hmap)/12)

    if (round(nside,1)!=nside) or ((nside%2)!=0):
        print ''
        print 'Not a standard Healpix map...'
        print ''
        exit()

    npix = 12*nside**2
    pix_area = 4.*np.pi / npix
    ncolumn = len(hmap)

    # get pixels in aperture
    phi   = lon*np.pi/180.
    theta = np.pi/2.-lat*np.pi/180.
    vec0  = hp.ang2vec(theta, phi)

    ## Get the pixels within the innermost (source) aperture
    innerpix = hp.query_disc(nside=nside, vec=vec0, radius=aper_inner_radius, nest=nested)

    #Get the background ring pixel numbers"
    outerpix1 = hp.query_disc(nside=nside, vec=vec0, radius=aper_outer_radius1, nest=nested)


    ## Get the pixels within the outer-ring of the background annulus
    outerpix2 = hp.query_disc(nside=nside, vec=vec0, radius=aper_outer_radius2, nest=nested)

    # Identify and remove the bad pixels
    # In this scheme, all of the bad pixels should have been labeled with HP.UNSEEN in the HEALPix maps
    bad0 = np.where(hmap[innerpix] == hp.UNSEEN)

    innerpix_masked = np.delete(innerpix,bad0)
    ninnerpix = len(innerpix_masked)

    bad1 = np.where(hmap[outerpix1] == hp.UNSEEN)
    outerpix1_masked = np.delete(outerpix1,bad1)
    nouterpix1 = len(outerpix1_masked)

    bad2 = np.where(hmap[outerpix2] == hp.UNSEEN)
    outerpix2_masked = np.delete(outerpix2,bad2)
    nouterpix2 = len(outerpix2_masked)

    if (ninnerpix == 0) or (nouterpix1 == 0) or (nouterpix2 == 0):
        print ''
        print '***No good pixels inside aperture!***'
        print ''
        fd = np.nan
        fd_err = np.nan
        exit()

    innerpix  = innerpix_masked
    outerpix1 = outerpix1_masked
    outerpix2 = outerpix2_masked

    # find pixels in the annulus (between outerradius1 and outeradius2)
    # In other words, remove pixels of Outer Radius 2 that are enclosed within Outer Radius 1
    bgpix = np.delete(outerpix2, outerpix1)

    nbgpix = len(bgpix)

    factor = convertToJy(units, thisfreq, npix)

    # Get pixel values in inner radius, converting to Jy/pix
    fd_jypix_inner = hmap[innerpix] * factor

    # sum up integrated flux in inner
    fd_jy_inner = np.sum(fd_jypix_inner)

    # same for outer radius but take a robust estimate and scale by area
    fd_jy_outer = np.median(hmap[bgpix]) * factor

    # subtract background
    fd_bg        = fd_jy_outer
    fd_bg_scaled = fd_bg*float(ninnerpix)
    fd           = fd_jy_inner - fd_bg_scaled

    
  
    
    if (noise_model == 0):

        pix_area = 4.*np.pi / npix
        
        Npoints = (pix_area*ninnerpix) /  (1.13*(float(aper_inner_radius)/60.*np.pi/180.)**2)
        
        Npoints_outer = (pix_area*nbgpix) /  (1.13*(float(aper_inner_radius)/60.*np.pi/180.)**2)

        fd_err = np.std(hmap[bgpix]) * factor * ninnerpix / math.sqrt(Npoints)


    if (noise_model == 1):

        #Robust sigma (median absolute deviation) noise model
        # works exactly for white uncorrelated noise only!
        #k = np.pi/2.

        # fd_err = factor * math.sqrt(float(ninnerpix) + (k * float(ninnerpix)**2/nbgpix)) * mad_std(hmap[bgpix])

        fd_err = factor *  mad_std(hmap[bgpix])

    return fd, fd_err, fd_bg_scaled


def haperfluxMany(inputlist, maplist, radius, rinner, router, galactic=True, decimal=True, noise_model=0):

    """
    Gets aperture results for a list of sources accross a list of maps using 'haperflux'



    Parameters
    ----------

    inputlist : File containing a list of paths of target coordinates
    maplist :  File containing a list of paths to HEALPix maps to be used
    radius : Radius of the source aperture in arcmin
    rinner : Radius of the inner boundary of the background annulus
    router : Radius of the outer boundary of the background annulus
    galactic :
    """

    ##  Names and frequencies of the sample maps included in this repo.

    freqlist =     ['30','44','70','100','143','217','353','545','857','1249','1874','2141','2998','3331','4612','4997','11992','16655','24983','24983','24983','33310']
    freqval =      [28.405889, 44.072241,70.421396,100.,143.,217.,353.,545.,857.,1249.,1874.,2141.,2141.,2998.,2998.,3331.,4612.,4997.,11992.,16655.,24983.,24983.,24983.,33310.]
    band_names =   ["akari9", "dirbe12","iras12","wise12","akari18","iras25","iras60","akari65","akari90","dirbe100","iras100","akari140","dirbe140","akari160","dirbe240","planck857", "planck545"]

    k0 = 1.0
    k1 = rinner/radius
    k2 = router/radius
    apcor = ((1 - (0.5)**(4*k0**2))-((0.5)**(4*k1**2) - (0.5)**(4*k2**2)))**(-1)

    # 'galactic' overrules 'decimal'
    if (galactic==True):
        dt=[('sname',np.dtype('S13')),('glon',np.float32),('glat',np.float32)]
        targets = np.genfromtxt(inputlist, delimiter=",",dtype=dt)

    ns = len(targets['glat'])

    fd3 = -1
    fd_err3 = -1

    fn = np.genfromtxt(maplist, delimiter=" ", dtype='str')
    nmaps = len(fn)
    ## Initialize the arrays which will hold the results
    fd_all = np.zeros((ns,nmaps))
    fd_err_all = np.zeros((ns,nmaps))
    fd_bg_all = np.zeros((ns,nmaps))

    # Start the actual processing: Read-in the maps.
    for ct2 in range(0,nmaps):
        xtmp_data, xtmp_head = hp.read_map(fn[ct2], h=True, verbose=False, nest=False)
        freq = dict(xtmp_head)['FREQ']
        units = dict(xtmp_head)['TUNIT1']
        freq_str = str(freq)
        idx = freqlist.index(str(freq))
        currfreq = int(freq)

        if (radius == None):
            radval = fwhmlist[idx]
        else:
            radval = radius


        for ct in range(0,ns):

            glon = targets['glon'][ct]
            glat = targets['glat'][ct]

            fd_all[ct,ct2], fd_err_all[ct,ct2], fd_bg_all[ct,ct2] = \
                haperflux(inmap= xtmp_data, freq= currfreq, lon=glon, lat=glat, aper_inner_radius=radius, aper_outer_radius1=rinner, \
                        aper_outer_radius2=router,units=units, noise_model=noise_model)

            if (np.isfinite(fd_err_all[ct,ct2]) == False):
                fd_all[ct,ct2] = -1
                fd_err_all[ct,ct2] = -1
            else:
                if radius==None:
                    fd_all[ct,ct2] = fd_all[ct,ct2]*apcor
                    fd_err_all[ct,ct2] = fd_err_all[ct,ct2]*apcor

    return fd_all, fd_err_all, fd_bg_all
