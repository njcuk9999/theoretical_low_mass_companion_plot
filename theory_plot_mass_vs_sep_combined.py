#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2016-11-08

@author: Neil Cook

Produces a plot in primary mass vs separation space for defined surveys/
telescopes

Version 0.0.001
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from astropy import units as u
import os
from tqdm import tqdm
from collections import OrderedDict

# =============================================================================
# Define variables
# =============================================================================
workspace = "/Astro/Projects/Fede_Work/"
savepath = workspace + 'Plots/literature_BD_desert/'

# define mass range (in Msun) and separation range (in log AU) to be considered
masslow = 0.01
masshigh = 0.22
logseplow = -2.0
logsephigh = 3.0

# --------------------------
# set up easy units
ums = u.m*u.s**-1
msun = u.Msun
mjup = u.Mjup

# define mass of the companion (if none use q)
dm2 = 40*u.Mjup
dm2 = None
# mass ratio of companion to primary (if not using a constant mass of companion)
dq = 1.0

# Distance to observation objects
ddist = 50*u.pc

# Other
asun = 4.0
csun = 1.0
aM = 2.3
cM = 0.23

psavename = 'Theory_plot_combined_q={0:.2f}_d={1:.2f}'.format(dq, ddist.value)

# -----------------------------------------------------------------------------
# define surveys/instruments to plot
#
# Keyword arguments that are required:
#
#  name      name of instrument
#  method    method ("Astrometric", "RV", "Transit", "Direct Imaging")
#  colour    colour of the instrument
#
# Once these are selected the following are needed (based on method)
#
# Astrometric: A, M2, period, dist, M0, mag0, a0, C0, dist0, mag1, a1, C1, dist1
#
# RV: K, M2, period, e, sini, M0, mag0, a0, C0, dist0, mag1, a1, C1, dist1
#
# Transit: M2, period, M0, mag0, a0, C0, dist0, mag1, a1, C1, dist1
#
# Direct Imaging: iwa1, dist, mag0, a0, C0, dist0, mag1, a1, C1, dist1, contrast
#                 iwa2 is optional for Direct imaging
#
# where:
#     a =       astropy quantity, Astrometric Amplitude
#               (astropy units of angle)
#     k =       astropy quantity, Radial velocity semi-amplitude
#               (astropy units of velocity)
#     m2 =      astropy quantity, Mass of companion (astropy units of mass)
#     period =  astropy quantity, Longest possible observation baseline
#               (astropy units of time)
#     e =       float, eccentricity of orbit (0 for circular, not units)
#     sini =    float, sin of the inclination angle
#               (1 for best possible, average = pi/4?)
#     iwa1 =   astropy quantity, inner working angle of telescope
#             (astropy units of angle)
#     iwa2 =   astropy quantity, outer working angle of telescope
#             (astropy units of angle) - optional None if no outer working angle
#     dist =    astropy quantity, distance to primary stars in question
#               (astropy units of distance)
#
#     m0 =      astropy quantity, mass of reference star (astropy units of mass)
#     mag0 =    float, magnitude of the reference star
#     a0 =      float, (no units) mass power law index
#               (such that L0 = C0*M0^(a0)) where L0 is luminosity in solar
#               units and M0 is mass in solar units
#     c0 =      float, (no units) mass power law coefficient
#               (such that L0 = C0*M0^(a0)) where L0 is luminosity in solar
#               units and M0 is mass in solar units
#     dist0 =   astropy quantity, distance to reference star
#               (astropy units of distance)
#     mag1 =    astropy quantity, magnitude limit of instrument
#               (in same magnitude regime as mag0)
#     a1 =      float, (no units) mass power law index for primary star
#               (such that L1 = C1*M1^(a1)) where L1 is luminosity in solar
#               units and M1 is mass in solar units
#     c1 =      float, (no units) mass power law coefficient
#               (such that L1 = C1*M0^(a1)) where L1 is luminosity in solar
#               units and M1 is mass in solar units
#     dist1 =   astropy quantity, distance to primary stars in question
#               (astropy units of distance)
#
#     contrast = float, delta magnitude between primary and secondary
#                ( in same magnitude regime as mag0 and mag1)

"""
basically for an astrometric instrument I would need:
- the astrometric amplitude sensitivity (in micro arcsce)
- the baseline for your observations
- the distance, mag, mass for a reference star observed with that instrument
- the distance of your objects, mag limit of instrument
"""

cat = dict()
cat['as:gaia'] = dict(name='Astrometric: Gaia', method='Astrometric',
                      color='b', zo=4, ls='--',
                      aamp=150*u.uas, m2=dm2, period=30*u.yr, dist=ddist,
                      m0=1*msun, mag0=8.406, a0=asun, c0=csun, dist0=44*u.pc,
                      mag1=16, a1=aM, c1=cM, dist1=ddist, q=dq)

cat['rv:crires'] = dict(name='RV: CRIRES/VLT', method='RV',
                     color='r', zo=3, ls='-',
                     kamp=5*ums, m2=dm2, period=5.8*u.yr, e=0, sini=1,
                     m0=1*msun, mag0=6.85, a0=asun, c0=csun, dist0=44*u.pc,
                     mag1=9.5, a1=aM, c1=cM, dist1=ddist, q=dq)

# cat['rv:sloan'] = dict(name='RV: Sloan 2.5m (APOGEE)', method='RV',
#                      color='r', zo=3, ls='-',
#                      kamp=100*ums, m2=dm2, period=3*u.yr, e=0, sini=1,
#                      m0=1*msun, mag0=7.27, a0=asun, c0=csun, dist0=44*u.pc,
#                      mag1=14, a1=aM, c1=cM, dist1=ddist)

cat['rv:hrs'] = dict(name='RV: HRS/Hobby-Eberly Telescope', method='RV',
                     color='m', zo=3, ls='-',
                     kamp=3*ums, m2=dm2, period=10*u.yr, e=0, sini=1,
                     m0=1 * msun, mag0=8.69, a0=asun, c0=csun, dist0=44 * u.pc,
                     mag1=10, a1=aM, c1=cM, dist1=ddist, q=dq)

cat['rv:spirou'] = dict(name='RV: SPiRou/CFHT', method='RV',
                        color='purple', zo=3, ls='--',
                        kamp=4*ums, m2=dm2, period=5*u.yr, e=0, sini=1,
                        m0=1 * msun, mag0=7.27, a0=asun, c0=csun,
                        dist0=44 * u.pc, mag1=12, a1=aM, c1=cM, dist1=ddist,
                        q=dq)

# cat['rv:rave'] = dict(name='RV: RAVE/UKST', method='RV',
#                       color='r', zo=3, ls='-',
#                       kamp=1500*ums, m2=dm2, period=10*u.yr, e=0, sini=1,
#                       m0=1 * msun, mag0=7.7, a0=asun, c0=csun, dist0=44 * u.pc,
#                       mag1=12, a1=aM, c1=cM, dist1=ddist)

cat['rv:carm'] = dict(name='RV: CARMENES/Calar Alto', method='RV',
                      color='r', zo=3, ls='--',
                      kamp=1*ums, m2=dm2, period=10*u.yr, e=0, sini=1,
                      m0=1 * msun, mag0=7.27, a0=asun, c0=csun, dist0=44 * u.pc,
                      mag1=9, a1=aM, c1=cM, dist1=ddist,
                        q=dq)

cat['tr:kepler'] = dict(name='Transit: Kepler', method='Transit',
                        color='lime', zo=1, ls='-',
                        m2=dm2, period=3.5*u.yr,
                        m0=1.3*msun, mag0=12.2, a0=asun, c0=csun,
                        dist0=516*u.pc, mag1=16, a1=aM, c1=cM, dist1=ddist,
                        q=dq)

cat['tr:k2'] = dict(name='Transit: Kepler 2', method='Transit',
                    color='darkgreen', zo=1, ls='-',
                    m2=dm2, period=80*u.day,
                    m0=1.3 * msun, mag0=12.2, a0=asun, c0=csun, dist0=516 * u.pc,
                    mag1=16, a1=aM, c1=cM, dist1=ddist,
                    q=dq)

# cat['tr:niriss soss'] = dict(name='Transit: NIRISS SOSS', method='Transit',
#                         color='darkgreen', zo=1, ls='--',
#                         m2=dm2, period=5*u.yr,
#                         m0=1.3 * msun, mag0=7.27, a0=asun, c0=csun,
#                         dist0=516 * u.pc, mag1=24, a1=aM, c1=cM, dist1=ddist)

# cat['stis'] = dict(name='STIS/Hubble', method='Direct Imaging',
#                    color='0.5', zo=2,
#                    iwa=0.4*u.arcsec, dist=ddist,
#                    m0=1*msun, mag0=8.69, a0=asun, c0=csun, dist0=44*u.pc,
#                    mag1=25.8, a1=aM, c1=cM, dist1=ddist, contrast=8.42)

cat['di:keck'] = dict(name='Direct Imaging: KPIC/Keck',
                      method='Direct Imaging', color='0.5', zo=3, ls='--',
                      iwa1=25 * u.mas, dist=ddist,
                      m0=1*msun, mag0=7.27, a0=asun, c0=csun, dist0=44*u.pc,
                      mag1=15.5, a1=aM, c1=cM, dist1=ddist, contrast=4.5)

cat['di:wfc3'] = dict(name='Direct Imaging: WFC3/Hubble',
                      method='Direct Imaging', color='k', zo=1, ls='-',
                      iwa1=0.425 * u.arcsec, dist=ddist,
                      m0=1*msun, mag0=7.27, a0=asun, c0=csun, dist0=44*u.pc,
                      mag1=27.3, a1=aM, c1=cM, dist1=ddist, contrast=4.5)

cat['di:jwstami'] = dict(name='Direct Imaging: NIRISS AMI/JWST',
                         method='Direct Imaging', color='orange', zo=2, ls='--',
                         iwa1=0.1*u.arcsec, iwa2=0.3*u.arcsec, dist=ddist,
                         m0=1*msun, mag0=7.27, a0=asun, c0=csun, dist0=44*u.pc,
                         mag1=35, a1=aM, c1=cM, dist1=ddist, contrast=4.5)

cat['di:jwstnircam'] = dict(name='Direct Imaging: NIRCam/JWST',
                            method='Direct Imaging', color='yellow', zo=2,
                            ls='--',
                            iwa1=0.4*u.arcsec, dist=ddist,
                            m0=1*msun, mag0=7.27, a0=asun, c0=csun,
                            dist0=44*u.pc, mag1=35, a1=aM, c1=cM, dist1=ddist,
                            contrast=4.5)

# cat['di:epics'] = dict(name='Direct Imaging: E-ELT/EPICS',
#                             method='Direct Imaging', color='salmon', zo=3,
#                             ls='--',
#                             iwa1=15*u.mas, dist=ddist,
#                             m0=1*msun, mag0=7.7, a0=asun, c0=csun,
#                             dist0=44*u.pc, mag1=9, a1=aM, c1=cM, dist1=ddist,
#                             contrast=3.3)

# =============================================================================
# Define functions
# =============================================================================
def sep_from_period(p, m1, m2):
    """
    Returns the Keplerian separation (in AU) for a companion with period p
    and primary mass m1 and companion mass m2

          |  p   |2/3       |  m1 + m2  |1/3
    sep = | ---- |      *   | --------- |
          | 1 yr |          |   1 Msun  |


    :param p: orbital period (with astropy units of time)
    :param m1: mass of star (with astropy units)
    :param m2: mass of companion (with astropy units)
    :return sep: star-companion distance (separation in astropy units of AU)
    """


    part1 = (p.to(u.year)/(1*u.year))**(2/3)
    part2 = ((m1.to(u.Msun) + m2.to(u.Msun))/(1*u.Msun))**(1/3)
    return (part1 * part2) * u.AU


def sep_from_k(k, m1, m2, e, sini):
    """
    Returns separation based on instrumentation (i.e. RV semi amplitude)

    sep = (k/(6.4 ms^-1)) * (sin(i)*((m2)/(10 Mearth))^2 *

          |    k    |   |  m2 * sin(i) |2    | m1 + m2 |^-1     | 1 yr |^(-2/3)
    sep = | ------- | * |  ----------- |  *  |---------|    *   | ---- |
          | 6.4 m/s |   |   10 Mearth  |     |  1 Msun |        | days |

            * (1 - e^2)^-1

    :param k:
    :param m1: mass of star (with astropy units)
    :param m2: mass of companion (with astropy units)
    :param e: eccentricity of the system (no units)
    :param sini: sin of the inclination of the system
    :return:
    """
    part1 = (k.to(u.m*u.s**(-1))/(6.4*u.m*u.s**-1))**(-2)
    part2 = (sini * (m2.to(u.Mearth)/(10*u.Mearth)))**2
    part3 = (1 - e**2)**(-1)
    part4 = ((m1.to(u.Msun) + m2.to(u.Msun))/(1*u.Msun))**(-1)
    part5 = (u.yr.to(u.day)) ** (-2/3)

    return part1 * part2 * part3 * part4 * part5 * u.AU


def sep_from_a(a, m1, m2, d):
    """
    Returns separation based on instrumentation
    (i.e. astrometric semi amplitude)

          |  a    |     |  m2     |-1      |  d   |     |  m1 + m2   |
    sep = | ----- |  *  | ------- |    *   | ---- |  *  | ---------- |
          | 3 mas |     |1 Mearth |        | 1 pc |     |   1 Msun   |


    :param a:
    :param m1: mass of star (with astropy units)
    :param m2: mass of companion (with astropy units)
    :param d: distance to star (with astropy units)
    :return:
    """
    part1 = a.to(u.uas)/(3*u.uas)
    part2 = (m2.to(u.Mearth)/(1*u.Mearth))**(-1)
    part3 = d.to(u.pc)/(1*u.pc)
    part4 = ((m1.to(u.Msun) + m2.to(u.Msun))/(1*u.Msun))

    return part1 * part2 * part3 * part4 * u.AU


def sep_from_imaging(iwa, dist):
    """
          | dist  |             Rad
    sep = | ----- |  *   tan(IWA   )
          | 1 AU  |

    :param iwa:
    :param dist:
    :return:
    """

    part1 = dist.to(u.AU)/(1*u.AU)
    part2 = np.tan(np.deg2rad(iwa.to(u.deg)))

    return part1 * part2


def masslimit_from_maglimit(m0, mag1, mag0, a1, a0, dist1, dist0, c1, c0):
    """

    :param m0: mass of reference star (astropy units of mass)
    :param mag1: magnitude limit of instrument
                 (in same magnitude regime as mag0)
    :param mag0: magnitude of the reference star
    :param a1: (no units) mass power law index for primary star
               (such that L1 = C1*M1^(a1)) where L1 is luminosity in solar
               units and M1 is mass in solar units
    :param a0: (no units) mass power law index
               (such that L0 = C0*M0^(a0)) where L0 is luminosity in solar
               units and M0 is mass in solar units
    :param dist1: distance to primary stars in question
                  (astropy units of distance)
    :param dist0: distance to reference star (astropy units of distance)
    :param c1: (no units) mass power law coefficient
                (such that L1 = C1*M0^(a1)) where L1 is luminosity in solar
                units and M1 is mass in solar units
    :param c0: (no units) mass power law coefficient
                (such that L0 = C0*M0^(a0)) where L0 is luminosity in solar
                units and M0 is mass in solar units
    :return:
    """
    part1 = 10**(-0.4*(mag1 - mag0))
    part2 = (m0.to(u.Msun)/(1*u.Msun))**(-a0)
    part3 = (dist1.to(u.m)/dist0.to(u.m))**2
    part4 = c0/c1

    return (part1 * part2 * part3 * part4) ** (1/a1) * u.Msun


def grid(x, y):
    xx = np.repeat(x, repeats=len(y)).reshape(len(x), len(y))
    yy = np.tile(y, reps=len(x)).reshape(len(x), len(y))
    zz = np.ones((len(x), len(y)), dtype=bool)
    return [xx, yy, zz]


def clabel(labels, values):
    l = ''
    for k in range(len(values)):
        if hasattr(values[k], 'unit'):
            args = [labels[k], values[k].value,
                    values[k].unit.to_string('latex')]
            l += '{0}={1:.2f} {2} '.format(*args)
        else:
            args = [labels[k], values[k]]
            l += '{0}={1:.2f} '.format(*args)
    return l


def RVpoly(m1, **kwargs):
    """
    Produce separations based on masses and keyword args for imaging
    :param m1:  numpy array, mass of the primary stars
    :param kwargs: kwarg args see below

    kwargs are:


    K =     astropy quantity, Radial velocity semi-amplitude
            (astropy units of velocity)

    M2 =    astropy quantity, Mass of companion (astropy units of mass)
    period = astropy quantity, Longest possible observation baseline
             (astropy units of time)
    e =     float, eccentricity of orbit (0 for circular, not units)
    sini =  float, sin of the inclination angle (1 for best possible,
            average = pi/4?)

    M0 =    astropy quantity, mass of reference star (astropy units of mass)
    mag0 =  float, magnitude of the reference star
    a0 =    float, (no units) mass power law index (such that L0 = C0*M0^(a0))
            where L0 is luminosity in solar units and M0 is mass in solar units
    C0 =    float, (no units) mass power law coefficient
            (such that L0 = C0*M0^(a0)) where L0 is luminosity in solar units
            and M0 is mass in solar units
    dist0 = astropy quantity, distance to reference star
            (astropy units of distance)
    mag1 =  astropy quantity, magnitude limit of instrument
            (in same magnitude regime as mag0)
    a1 = f  loat, (no units) mass power law index for primary star
            (such that L1 = C1*M1^(a1)) where L1 is luminosity in solar units
            and M1 is mass in solar units
    C1 =    float, (no units) mass power law coefficient
            (such that L1 = C1*M0^(a1)) where L1 is luminosity in solar units
            and M1 is mass in solar units
    dist1 = astropy quantity, distance to primary stars in question
            (astropy units of distance)


    :return:
    """
    # direct RV params mass of the companion, orbital period of the companion
    # eccentricity of the companion and sini of the companion
    k = kwargs.get("kamp", 10.0*u.m*u.s**-1)
    m2 = kwargs.get("m2", 40.0*u.Mjup)
    if m2 is None:
        q = kwargs.get("q", 1.0)
        m2 = q * m1
    period = kwargs.get("period", 5.0*u.year)
    e = kwargs.get("e", 0.0)
    sini = kwargs.get("sini", 1.0)
    # mass of the reference star, magnitude of reference star, for
    # mass-luminosity index "a0" const "C0" at a distance of dist0
    m0 = kwargs.get("m0", 1.0*u.Msun)
    mag0 = kwargs.get("mag0", -26.0)
    a0 = kwargs.get("a0", 4)
    c0 = kwargs.get("c0", 1)
    dist0 = kwargs.get("dist0", 1*u.AU)
    # limiting magnitude of instruments for star of mass-luminosity index "a1"
    # const "C1" at a distance of dist1
    mag1 = kwargs.get("mag1", 14)
    a1 = kwargs.get("a1", 2.3)
    c1 = kwargs.get("c1", 0.23)
    dist1 = kwargs.get("dist1", 100*u.pc)

    # sep less than period sep
    sep1 = sep_from_period(period, m1, m2)

    # sep less than K sep
    sep2 = sep_from_k(k, m1, m2, e, sini)

    # mass greater than masslimit
    mass1 = masslimit_from_maglimit(m0, mag1, mag0, a1, a0,
                                    dist1, dist0, c0, c1)

    # make vertices

    # first deal with line 1
    mmask = m1 > mass1
    vx = (m1[m1 > mass1]).value
    vy = (np.min([(sep1[mmask]).value, (sep2[mmask]).value], axis=0))
    # now deal with line 2
    vx = np.append(vx, np.max(m1).value)
    vy = np.append(vy, 10**logseplow)
    # now deal with line 3
    vx = np.append(vx, mass1.value)
    vy = np.append(vy, 10**logseplow)
    # return vertices as polygon
    vx = np.append(vx, vx[0])
    vy = np.append(vy, vy[0])
    # return vertices as polygon
    return list(zip(vx, vy))


def Astropoly(m1, **kwargs):
    """
    Produce separations based on masses and keyword args for imaging
    :param m1:  numpy array, mass of the primary stars
    :param kwargs: kwarg args see below

    kwargs are:

    aamp =     astropy quantity, Astrometric Amplitude (astropy units of angle)
    m2 =    astropy quantity, Mass of companion (astropy units of mass)
    period = astropy quantity, Longest possible observation baseline
             (astropy units of time)
    dist =  astropy quantity, distance to primary stars in question
            (astropy units of distance)

    m0 =    astropy quantity, mass of reference star (astropy units of mass)
    mag0 =  float, magnitude of the reference star
    a0 =    float, (no units) mass power law index (such that L0 = C0*M0^(a0))
            where L0 is luminosity in solar units and M0 is mass in solar units
    c0 =    float, (no units) mass power law coefficient
            (such that L0 = C0*M0^(a0)) where L0 is luminosity in solar units
            and M0 is mass in solar units
    dist0 = astropy quantity, distance to reference star
            (astropy units of distance)
    mag1 =  astropy quantity, magnitude limit of instrument
            (in same magnitude regime as mag0)
    a1 = f  loat, (no units) mass power law index for primary star
            (such that L1 = C1*M1^(a1)) where L1 is luminosity in solar units
            and M1 is mass in solar units
    c1 =    float, (no units) mass power law coefficient
            (such that L1 = C1*M0^(a1)) where L1 is luminosity in solar units
            and M1 is mass in solar units
    dist1 = astropy quantity, distance to primary stars in question
            (astropy units of distance)


    :return:
    """
    # direct RV params mass of the companion, orbital period of the companion
    # eccentricity of the companion and sini of the companion
    aamp = kwargs.get("aamp", 150*u.uas)
    m2 = kwargs.get("m2", 40.0*u.Mjup)
    if m2 is None:
        q = kwargs.get("q", 1.0)
        m2 = q * m1
    period = kwargs.get("period", 5.0*u.year)
    dist = kwargs.get("dist", 100*u.pc)
    # mass of the reference star, magnitude of reference star, for
    # mass-luminosity index "a0" const "C0" at a distance of dist0
    m0 = kwargs.get("m0", 1.0*u.Msun)
    mag0 = kwargs.get("mag0", -26.0)
    a0 = kwargs.get("a0", 4)
    c0 = kwargs.get("c0", 1)
    dist0 = kwargs.get("dist0", 1*u.AU)
    # limiting magnitude of instruments for star of mass-luminosity index "a1"
    # const "C1" at a distance of dist1
    mag1 = kwargs.get("mag1", 14)
    a1 = kwargs.get("a1", 2.3)
    c1 = kwargs.get("c1", 0.23)
    dist1 = kwargs.get("dist1", 100*u.pc)


    # sep less than period sep
    sep1 = sep_from_period(period, m1, m2)

    # sep less than K sep
    sep2 = sep_from_a(aamp, m1, m2, dist)

    # mass greater than masslimit
    mass1 = masslimit_from_maglimit(m0, mag1, mag0, a1, a0,
                                    dist1, dist0, c0, c1)

    # make vertices

    # first deal with line 1
    mmask = m1 > mass1

    smask = sep1 > sep2

    vx = (m1[mmask & smask]).value
    vy = (sep1[mmask & smask]).value
    # now deal with line 3
    vx = np.append(vx, (m1[mmask & smask][::-1]).value)
    vy = np.append(vy, (sep2[mmask & smask][::-1]).value)
    # return vertices as polygon
    vx = np.append(vx, vx[0])
    vy = np.append(vy, vy[0])
    # return vertices as polygon
    return list(zip(vx, vy))



def Transpoly(m1, **kwargs):
    """
    Produce separations based on masses and keyword args for imaging
    :param m1:  numpy array, mass of the primary stars
    :param kwargs: kwarg args see below

    kwargs are:

    M2 =    astropy quantity, Mass of companion (astropy units of mass)
    period = astropy quantity, Longest possible observation baseline
             (astropy units of time)

    M0 =    astropy quantity, mass of reference star (astropy units of mass)
    mag0 =  float, magnitude of the reference star
    a0 =    float, (no units) mass power law index (such that L0 = C0*M0^(a0))
            where L0 is luminosity in solar units and M0 is mass in solar units
    C0 =    float, (no units) mass power law coefficient
            (such that L0 = C0*M0^(a0)) where L0 is luminosity in solar units
            and M0 is mass in solar units
    dist0 = astropy quantity, distance to reference star
            (astropy units of distance)
    mag1 =  astropy quantity, magnitude limit of instrument
            (in same magnitude regime as mag0)
    a1 = f  loat, (no units) mass power law index for primary star
            (such that L1 = C1*M1^(a1)) where L1 is luminosity in solar units
            and M1 is mass in solar units
    C1 =    float, (no units) mass power law coefficient
            (such that L1 = C1*M0^(a1)) where L1 is luminosity in solar units
            and M1 is mass in solar units
    dist1 = astropy quantity, distance to primary stars in question
            (astropy units of distance)


    :return:
    """
    # direct RV params mass of the companion, orbital period of the companion
    # eccentricity of the companion and sini of the companion
    m2 = kwargs.get("m2", 40.0*u.Mjup)
    if m2 is None:
        q = kwargs.get("q", 1.0)
        m2 = q * m1
    period = kwargs.get("period", 5.0*u.year)
    # mass of the reference star, magnitude of reference star, for
    # mass-luminosity index "a0" const "C0" at a distance of dist0
    m0 = kwargs.get("m0", 1.0*u.Msun)
    mag0 = kwargs.get("mag0", -26.0)
    a0 = kwargs.get("a0", 4)
    c0 = kwargs.get("c0", 1)
    dist0 = kwargs.get("dist0", 1*u.AU)
    # limiting magnitude of instruments for star of mass-luminosity index "a1"
    # const "C1" at a distance of dist1
    mag1 = kwargs.get("mag1", 14)
    a1 = kwargs.get("a1", 2.3)
    c1 = kwargs.get("c1", 0.23)
    dist1 = kwargs.get("dist1", 100*u.pc)


    # sep less than period sep
    sep1 = sep_from_period((1/3) * period, m1, m2)

    # mass greater than masslimit
    mass1 = masslimit_from_maglimit(m0, mag1, mag0, a1, a0,
                                    dist1, dist0, c0, c1)
    seps = [F_all(np.min, u.AU * 10**logseplow, sep1),
            F_all(np.max, sep1, u.AU * 10**logsephigh)]
    ms = np.repeat(mass1, len(seps))

    label1 = 'Transit ' + clabel(['P'], [period])
    label3 = 'Transit ' + clabel(['Mag limit', 'D'], [mag1, dist1])

    # test
    # plt.plot(m1, sep1, c='r', label='Period limit: ' + label1)
    # plt.plot(m1, sep2, c='b', label='Astrometry limit: ' + label2)
    # plt.plot(ms, seps, c='g', label='Bright limit: ' + label3)
    # plt.legend(loc=0)
    # plt.yscale('log')
    # plt.show()
    # plt.close()

    # define vertices

    # first deal with line 1
    mmask = m1 > mass1
    vx = (m1[m1 > mass1]).value
    vy = (sep1[mmask]).value
    # now deal with line 2
    vx = np.append(vx, np.max(m1).value)
    vy = np.append(vy, 10**logseplow)
    # now deal with line 3
    vx = np.append(vx, mass1.value)
    vy = np.append(vy, 10**logseplow)
    # return vertices as polygon
    vx = np.append(vx, vx[0])
    vy = np.append(vy, vy[0])
    # return vertices as polygon
    return list(zip(vx, vy))


def Imagingpoly(m1, **kwargs):
    """
    Produce separations based on masses and keyword args for imaging
    :param m1:  numpy array, mass of the primary stars
    :param kwargs: kwarg args see below

    kwargs are:

    iwa1 =   astropy quantity, inner working angle of telescope
            (astropy units of angle)
    iwa2 =   astropy quantity, outer working angle of telescope
            (astropy units of angle) - optional None if no outer working angle
    dist =  astropy quantity, distance to primary stars in question
            (astropy units of distance)

    M0 =    astropy quantity, mass of reference star (astropy units of mass)
    mag0 =  float, magnitude of the reference star
    a0 =    float, (no units) mass power law index (such that L0 = C0*M0^(a0))
            where L0 is luminosity in solar units and M0 is mass in solar units
    C0 =    float, (no units) mass power law coefficient
            (such that L0 = C0*M0^(a0)) where L0 is luminosity in solar units
            and M0 is mass in solar units
    dist0 = astropy quantity, distance to reference star
            (astropy units of distance)
    mag1 =  astropy quantity, magnitude limit of instrument
            (in same magnitude regime as mag0)
    a1 = f  loat, (no units) mass power law index for primary star
            (such that L1 = C1*M1^(a1)) where L1 is luminosity in solar units
            and M1 is mass in solar units
    C1 =    float, (no units) mass power law coefficient
            (such that L1 = C1*M0^(a1)) where L1 is luminosity in solar units
            and M1 is mass in solar units
    dist1 = astropy quantity, distance to primary stars in question
            (astropy units of distance)

    contrast = float, delta magnitude between primary and secondary
               (in same magnitude regime as mag0 and mag1)


    :return:
    """
    # direct RV params mass of the companion, orbital period of the companion
    # eccentricity of the companion and sini of the companion
    iwa1 = kwargs.get("iwa1", 0.1 * u.arcsec)
    iwa2 = kwargs.get("iwa2", None)
    dist = kwargs.get("dist", 100 * u.pc)
    # mass of the reference star, magnitude of reference star, for
    # mass-luminosity index "a0" const "C0" at a distance of dist0
    m0 = kwargs.get("m0", 1.0 * u.Msun)
    mag0 = kwargs.get("mag0", -26.0)
    a0 = kwargs.get("a0", 4)
    c0 = kwargs.get("c0", 1)
    dist0 = kwargs.get("dist0", 1 * u.AU)
    # limiting magnitude of instruments for star of mass-luminosity index "a1"
    # const "C1" at a distance of dist1
    mag1 = kwargs.get("mag1", 14)
    a1 = kwargs.get("a1", 2.3)
    c1 = kwargs.get("c1", 0.23)
    dist1 = kwargs.get("dist1", 100 * u.pc)
    contrast = kwargs.get("contrast", 0)


    # sep less than period sep
    sep_limit = sep_from_imaging(iwa1, dist)
    if iwa2 is not None:
        sep_limit2 = sep_from_imaging(iwa2, dist)
    # sep1 = np.repeat(sep_limit, len(m1))

    # mass greater than masslimit
    mass1 = masslimit_from_maglimit(m0, mag1 - contrast, mag0, a1, a0,
                                    dist1, dist0, c0, c1)

    # define vertices

    # first deal with line 1
    vx = [mass1.value, np.max(m1).value]
    vy = [sep_limit.value, sep_limit.value]
    # now deal with line 2
    vx = np.append(vx, np.max(m1).value)
    if iwa2 is None:
        vy = np.append(vy, 10 ** logsephigh)
    else:
        vy = np.append(vy, sep_limit2.value)
    # now deal with line 3
    vx = np.append(vx, mass1.value)
    if iwa2 is None:
        vy = np.append(vy, 10 ** logsephigh)
    else:
        vy = np.append(vy, sep_limit2.value)
    vx = np.append(vx, vx[0])
    vy = np.append(vy, vy[0])

    # return vertices as polygon
    return list(zip(vx, vy))


def F_all(F, *args):
    array = []
    for arg in args:
        array = np.append(array, np.array(arg).flatten())
    return F(array)


def test_plot(frame, cc, s):
    if len(s) > 0:
        frame.plot(s[0][0], s[0][1], c=cc, ls='-', lw=2, label=s[0][2])
    if len(s) > 1:
        frame.plot(s[1][0], s[1][1], c=cc, ls='--', lw=2, label=s[1][2])
    if len(s) > 2:
        frame.plot(s[2][0], s[2][1], c=cc, ls=':', lw=2, label=s[2][2])

    frame.set_yscale('log')
    frame.set_xlabel('Primary mass / $M_{odot}$')
    frame.set_ylabel('Separation/ AU')
    frame.legend(loc=0, fontsize='small')
    frame.set_xlim(masslow, masshigh)
    frame.set_ylim(10**logseplow)
    return frame


# =============================================================================
# Start of code
# =============================================================================



# Main code here
# -------------------------------------------------------------------------
if __name__ == "__main__":
    seps = u.AU * 10 ** np.linspace(logseplow, logsephigh, 1000)
    masses = u.Msun * np.linspace(masslow, masshigh, 1000)


    # set up figure
    plt.close()
    handles, labels = [], []
    fig, frame = plt.subplots(1, 1)
    fig.set_size_inches(7, 6)

    ocat = OrderedDict(sorted(cat.items()))

    for catkey in ocat:

        name = cat[catkey]['name']
        method = cat[catkey]['method']
        colour = cat[catkey]['color']
        ls = cat[catkey]['ls']
        zorder = cat[catkey]['zo']
        keyargs = cat[catkey]

        # Get separations and masses based on kwargs
        if method == 'RV':
            # Deal with RV
            polygon = RVpoly(masses, **keyargs)
        elif method == 'Astrometric':
            # Deal with Astrometry
            polygon = Astropoly(masses, **keyargs)
        elif method == 'Transit':
            # Deal with Transits
            polygon = Transpoly(masses, **keyargs)
        elif method == 'Direct Imaging':
            # Deal with direct imaging
            polygon = Imagingpoly(masses, **keyargs)
        else:
            continue

        patch = patches.PathPatch(Path(polygon), facecolor=colour, lw=1,
                                  alpha=0.15, zorder=zorder)
        frame.add_patch(patch)
        patch = patches.PathPatch(Path(polygon), facecolor='None', lw=1.5,
                                  ls=ls, edgecolor=colour, alpha=1,
                                  zorder=zorder)
        frame.add_patch(patch)

        frame.set_xlim(masslow, masshigh)
        frame.set_ylim(10**logseplow, 10**logsephigh)

        frame.set_yscale('log')

        handle1 = patches.Patch(edgecolor=colour, facecolor='none',
                               label=name,ls=ls)
        handle2 = patches.Patch(color=colour, alpha=0.25, label=name)
        handles.append((handle1, handle2)), labels.append(name)


    frame.set_xlabel('Primary mass [$M_{\odot}$]')
    frame.set_ylabel('Separation [AU]')
    frame.legend(handles, labels, loc=8, numpoints=1, scatterpoints=1,
                 ncol=2, bbox_to_anchor=(0.5, -0.4), fontsize='small')

    # save show and close
    print('\n Saving graph...')
    plt.savefig(savepath + psavename + '.png', bbox_inches='tight', dpi=300)
    plt.savefig(savepath + psavename + '.ps')
    plt.savefig(savepath + psavename + '.pdf', bbox_inches='tight')
    plt.close()



# =============================================================================
# End of code
# =============================================================================
