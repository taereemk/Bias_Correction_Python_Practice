# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 17:13:17 2025

@author: tk6869
"""
# Function for bias correction

# The original code is implemented in the downscaleR R package, 
# developed by the Santander Meteorology Group (Bedia et al., 2020; Iturbide et al., 2019).
# We have converted the original code to Python.

import numpy as np
from scipy.stats import gamma
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import gamma, norm
from scipy.optimize import curve_fit, root
from scipy import stats

def eqm(o, p, s, precip=True, pr_threshold=0.1, n_quantiles=None, extrapolation='constant'):
    o = o[~np.isnan(o)]
    p = p[~np.isnan(p)]
    s = s[~np.isnan(s)]
    
    # Precipitation case
    if precip:
        threshold = pr_threshold
        if len(o) > 0:
            # Adjust precipitation frequency
            params = adjust_precip_freq(o, p, threshold)
            p = params['p']
            nP = params['nP']
            Pth = params['Pth']
        else:
            nP = None

        smap = np.full(len(s), np.nan)
        if len(p) > 0 and len(o) > 0:
            if np.sum(p > Pth) > 0:
                rain = np.where((s > Pth) & ~np.isnan(s))[0]
                noRain = np.where((s <= Pth) & ~np.isnan(s))[0]
                drizzle = np.where((s > Pth) & (s <= np.min(p[p > Pth], axis=0)) & ~np.isnan(s))[0]

                if len(rain) > 0:
                    # Calculate empirical CDF
                    eFrc = ECDF(s[rain])

                    if n_quantiles is None:
                        n_quantiles = len(p)
                    bins = n_quantiles
                    qo = np.percentile(o[o > threshold], np.linspace(1 / bins, 1 - 1 / bins, bins) * 100)
                    qp = np.percentile(p[p > Pth], np.linspace(1 / bins, 1 - 1 / bins, bins) * 100)

                    # Approximating function
                    p2o = np.interp(s[rain], qp, qo)

                    smap[rain] = p2o

                    if extrapolation == 'constant':
                        smap[rain][s[rain] > np.max(qp)] = smap[rain][s[rain] > np.max(qp)] + (qo[-1] - qp[-1])
                        smap[rain][s[rain] < np.min(qp)] = smap[rain][s[rain] < np.min(qp)] + (qo[0] - qp[0])
                    else:
                        smap[rain][s[rain] > np.max(qp)] = qo[-1]
                        smap[rain][s[rain] < np.min(qp)] = qo[0]

                else:
                    smap = np.zeros(len(s))
                    print("No precipitation days for the selected window.")
                    return smap

                if len(drizzle) > 0:
                    # Separate conditions for drizzle
                    condition1 = (s > np.min(p[p > Pth]))  # Condition for s > min(p[p > Pth])
                    condition2 = ~np.isnan(s)  # Condition for non-NaN s values
                    
                    # Combine conditions with logical AND
                    valid_condition = condition1 & condition2
                    
                    smap[drizzle] = np.percentile(s[valid_condition], eFrc(s[drizzle]))

                smap[noRain] = 0
            else:
                smap = s
                print("No rainy days in prediction. Bias correction not applied.")
        else:
            smap = s
    else:
        # Non-precipitation case
        if len(o) == 0 or len(p) == 0:
            smap = np.full(len(s), np.nan)
        else:
            if n_quantiles is None:
                n_quantiles = len(p)
            bins = n_quantiles
            qo = np.percentile(o, np.linspace(1 / bins, 1 - 1 / bins, bins) * 100)
            qp = np.percentile(p, np.linspace(1 / bins, 1 - 1 / bins, bins) * 100)

            p2o = np.interp(s, qp, qo)

            smap = p2o

            if extrapolation == 'constant':
                smap[s > np.max(qp)] = smap[s > np.max(qp)] + (qo[-1] - qp[-1])
                smap[s < np.min(qp)] = smap[s < np.min(qp)] + (qo[0] - qp[0])
            else:
                smap[s > np.max(qp)] = qo[-1]
                smap[s < np.min(qp)] = qo[0]

    return smap

# Adjust precipitation frequency
def adjust_precip_freq(obs, pred, threshold):
    o = obs[~np.isnan(obs)]
    p = pred[~np.isnan(pred)]
    nPo = np.sum(o < threshold)
    nPp = int(np.ceil(len(p) * nPo / len(o)))
    ix = np.argsort(p)
    Ps = np.sort(p)
    Pth = np.max(Ps[nPp:nPp + 2])
    inddrzl = np.where(Ps[nPp:] < threshold)[0]

    if len(inddrzl) > 0:
        Os = np.sort(o)
        indO = int(np.ceil(len(Os) * (nPp + len(inddrzl)) / len(Ps)))
        auxOs = Os[nPo:indO]
        if len(np.unique(auxOs)) > 6:
            # Fitting Gamma distribution
            auxGamma = gamma.fit(auxOs)
            Ps[nPp:nPp + len(inddrzl)] = gamma.rvs(auxGamma[0], loc=auxGamma[1], scale=auxGamma[2], size=len(inddrzl))
        else:
            Ps[nPp:nPp + len(inddrzl)] = np.mean(auxOs)
        Ps = np.sort(Ps)

    if nPo > 0:
        ind = min(nPp, len(p))
        Ps[:ind] = 0
    p[ix] = Ps
    pred[~np.isnan(pred)] = p
    return {'nP': [nPo, nPp], 'Pth': Pth, 'p': pred}

def scaling(o, p, s, scaling_type):
    if scaling_type == "additive":
        return s - np.mean(p) + np.mean(o)
    elif scaling_type == "multiplicative":
        return (s / np.mean(p)) * np.mean(o)  
    
        