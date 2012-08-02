'''Compute beta parameters.

The so-called beta parameters were developed in SNO as a measure of event
isotropy:

    The lth beta parameter, beta_l, is defined as the average value of the
    Legendre polynomial, P_l, of the cosine of the angle between each pair
    PMT hits in the event.

        beta_l = <P_l(cos(theta_ik)> where i != k

    Again, the angle is taken with respect to the fitted vertex position.
    The combination beta_14 = beta_1 + 4 * beta_4 was selected by the SNO
    collaboration for use in signal extraction due to the good separability
    it provides and the ease of parameterisation of the Gaussian-like
    distribution.

    - Measurement of the 8B Solar Neutrino Energy Spectrum at the Sudbury
      Neutrino Observatory, Jeanne R. Wilson, p. 179 (Ph.D. Thesis)
'''

from math import sqrt, acos
import numpy as np
from scipy.special import legendre
from rat import ROOT, dsreader
from pmtpos import pmtpos

ROOT.gROOT.SetBatch(True)

debug = False

try:
    profile_if_possible = profile
except NameError:
    profile_if_possible = lambda x: x

@profile_if_possible
def get_theta(A, B, C):
    c = sqrt(sum((A-B)**2))
    a = sqrt(sum((C-B)**2))
    b = sqrt(sum((C-A)**2))

    return acos(-0.5 * (c**2 - a**2 - b**2) / (a * b))

@profile_if_possible
def calculate_betas(ev):
    fit_position = np.array(ev.GetFitResult('scintFitter').GetVertex(0).GetPosition())
    hit_pmts = [ev.GetPMTUnCal(i) for i in range(ev.GetPMTUnCalCount())]

    npairs = len(hit_pmts) * (len(hit_pmts) - 1) / 2
    triangles = np.empty(shape=(npairs,3,3), dtype=np.float32)

    count = 0
    thetas = []
    for i, u in enumerate(hit_pmts[:-1]):
        if count % 200 == 0 and debug:
            print 'Pair', count, '/', npairs

        for v in hit_pmts[i+1:]:
            thetas.append(get_theta(pmtpos[u.GetID()], pmtpos[v.GetID()], fit_position))
            count += 1

    betas = {}
    beta14 = np.empty_like(thetas)
    for l in range(5):
        ps = legendre(l)(thetas)
        betas[l] = np.mean(ps)

        if l == 1:
            beta14 += ps
        elif l == 4:
            beta14 += 4.0 * ps

    betas['14'] = np.mean(beta14)

    return betas

