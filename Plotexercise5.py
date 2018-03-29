# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 12:41:47 2018

@author: Marcus Karlsson
"""

# Import necessary functions and libraries
from matplotlib import pyplot as mp
from matplotlib import mlab as mlab
import numpy as np
import math as math
import pylab as pylab
import scipy as sp
from scipy.stats import norm
from astropy.modeling.models import Moffat1D
from scipy.optimize import curve_fit

# Define constants (mean, FWHM, sigma, noise)
#sigma = fwhm / (2.0*math.sqrt(2.0 * math.log(2.0)))
#sigma3 = fwhm3 / (2.0 * math.sqrt(2.0 ** (1.0 / beta) - 1.0))


# Define how many data points you wish to work with
points = 100

# Define the mean and standard deviation of the noise
noise_mean = 0.0
noise_stdv_low = 0.0
noise_stdv_high = 0.1
noise_stdv = np.linspace(noise_stdv_low, noise_stdv_high, points)

# Define which values you wish to create a distribution
# of number between in the line profiles
number_low = -5.0
number_high = 5.0

# Defines the amplitude of the line profile samples, the mean
# from the low and high x-range, the standard deviation and
# the beta for the Moffat profile
Amp = 1.0
Mean = (number_low + number_high)/2
Stdv = 1.0
Beta = 1.0

# Creates a linear range between these values
# with the defined number of data points
range1 = np.linspace(number_low, number_high, points)

# Creates a range of values used for different standard
# deviation of noise with input (Start, End, Steps)
range2 = np.linspace(noise_stdv_low, noise_stdv_high, points)

# Creates a list of noise arrays with input
# (Mean, Standard deviation, Number of elements)
noise1 = [np.random.normal(noise_mean, noise_stdv[i], points) for i in range(points)]


# Fit data to a Gaussian, Moffat, and Sinc distribution

def gaussian(x, mu, sigma):
    return Amp * np.exp(-(x - mu)**2 / (2*(sigma)**2))

def moffat(x, mu2, sigma2):
    return Amp * (((x - mu2) / (sigma2)) ** 2.0 + 1.0) ** (-Beta)

def sinc(x, mu3, sigma3):
    return Amp * pylab.sin(math.pi * (x - mu3) * sigma3) / (math.pi * (x - mu3) * sigma3)

# Generate an list of 100 equal arrays of 100 random values from a
# Gaussian, Moffat, or Sinc distribution

G_sample1 = gaussian(range1, Mean, Stdv)
G_sample2 = []
for i in range(points):
    G_sample2.append(G_sample1)

M_sample1 = moffat(range1, Mean, Stdv)
M_sample2 = []
for i in range(points):
    M_sample2.append(M_sample1)

S_sample1 = sinc(range1, Mean, Stdv)
S_sample2 = []
for i in range(points):
    S_sample2.append(S_sample1)

# Add the generated noise at 100 different amplitude levels to
# the 100 arrays of gaussian, moffat, and sinc values

G_noise = [G_sample2[i] + noise1[i] for i in range(points)]
M_noise = [M_sample2[i] + noise1[i] for i in range(points)]
S_noise = [S_sample2[i] + noise1[i] for i in range(points)]

# Define the initial values of the amp, mean, and stdv the fit should
# guess the fitted values from
G_initial = [Mean, Stdv]
M_initial = [Mean, Stdv]
S_initial = [Mean, Stdv]

# Create a fit to the parameters using 'curve_fit' which generates a
# tuple array consisting of a fit to [amp, mean, stdv, (beta)] and the covariance
G_param1 = [curve_fit(gaussian, range1, G_noise[i], p0 = G_initial) for i in range(points)]
M_param1 = [curve_fit(moffat, range1, M_noise[i], p0 = M_initial) for i in range(points)]
S_param1 = [curve_fit(sinc, range1, S_noise[i], p0 = S_initial) for i in range(points)]

# Splitting up the parameters and the covariance into two arrays
G_param2, G_covar = map(list, zip(*G_param1))
M_param2, M_covar = map(list, zip(*M_param1))
S_param2, S_covar = map(list, zip(*S_param1))

# Extracting the fit to the mean parameters of the 100 arrays
G_param_mean = [i for i,_ in G_param2]
M_param_mean = [i for i,_ in M_param2]
S_param_mean = [i for i,_ in S_param2]

# Extracting the fit to the stdv parameters of the 100 arrays 
G_param_stdv = [i for _,i in G_param2]
M_param_stdv = [i for _,i in M_param2]
S_param_stdv = [i for _,i in S_param2]

# Creates two arrays to see how much the fit to the mean and the stdv
# differentiates from the actual mean and stdv of the values
# Mean = 0, Standard deviation = 1
M_diff_gauss = abs(np.linspace(Mean, Mean, points) - G_param_mean)
M_diff_moffat = abs(np.linspace(Mean, Mean, points) - M_param_mean)
M_diff_sinc = abs(np.linspace(Mean, Mean, points) - S_param_mean)
S_diff_gauss = abs(np.linspace(Stdv, Stdv, points) - G_param_stdv)
S_diff_moffat = abs(np.linspace(Stdv, Stdv, points) - M_param_stdv)
S_diff_sinc = abs(np.linspace(Stdv, Stdv, points) - S_param_stdv)

# Creates a linear fit to how the difference in fitted and actual
# mean and stdv changes with different amplitude of noise
M_line_gauss = np.polyfit(range2, M_diff_gauss, 1)
M_line_moffat = np.polyfit(range2, M_diff_moffat, 1)
M_line_sinc = np.polyfit(range2, M_diff_sinc, 1)
S_line_gauss = np.polyfit(range2, S_diff_gauss, 1)
S_line_moffat = np.polyfit(range2, S_diff_moffat, 1)
S_line_sinc = np.polyfit(range2, S_diff_sinc, 1)

# Plots the variation in fitted mean from actual mean = 0 as function
# of increasing noise amplitude and a linear fit to the values
mp.plot(range2, M_diff_gauss, "red", linewidth = 2)
mp.plot(range2, M_line_gauss[1] + range2 * M_line_gauss[0], "blue", linewidth = 2)
mp.title("Difference in mean with increasing noise (Gaussian)")
mp.xlabel("Standard deviation of noise")
mp.ylabel("Distance from actual mean = 0")
mp.show()

mp.plot(range2, M_diff_moffat, "red", linewidth = 2)
mp.plot(range2, M_line_moffat[1] + range2 * M_line_moffat[0], "blue", linewidth = 2)
mp.title("Difference in mean with increasing noise (Moffat)")
mp.xlabel("Standard deviation of noise")
mp.ylabel("Distance from actual mean = 0")
mp.show()

mp.plot(range2, M_diff_sinc, "red", linewidth = 2)
mp.plot(range2, M_line_sinc[1] + range2 * M_line_sinc[0], "blue", linewidth = 2)
mp.title("Difference in mean with increasing noise (Sinc)")
mp.xlabel("Standard deviation of noise")
mp.ylabel("Distance from actual mean = 0")
mp.show()

# Plots the variation in fitted stdv from actual stdv = 1 as function
# of increasing noise amplitude and a linear fit to the values
mp.plot(range2, S_diff_gauss, "blue", linewidth = 2)
mp.plot(range2, S_line_gauss[1] + range2 * S_line_gauss[0], "red", linewidth = 2)
mp.title("Difference in standard deviation with increasing noise (Gaussian)")
mp.xlabel("Standard deviation of noise")
mp.ylabel("Distance from actual stdv = 1")
mp.show()

mp.plot(range2, S_diff_moffat, "blue", linewidth = 2)
mp.plot(range2, S_line_moffat[1] + range2 * S_line_moffat[0], "red", linewidth = 2)
mp.title("Difference in standard deviation with increasing noise (Moffat)")
mp.xlabel("Standard deviation of noise")
mp.ylabel("Distance from actual stdv = 1")
mp.show()

mp.plot(range2, S_diff_sinc, "blue", linewidth = 2)
mp.plot(range2, S_line_sinc[1] + range2 * S_line_sinc[0], "red", linewidth = 2)
mp.title("Difference in standard deviation with increasing noise (Sinc)")
mp.xlabel("Standard deviation of noise")
mp.ylabel("Distance from actual stdv = 1")
mp.show()


# Parameter fit standard deviations:

G_mean_stdv = np.std(G_param_mean)
G_stdv_stdv = np.std(G_param_stdv)

M_mean_stdv = np.std(M_param_mean)
M_stdv_stdv = np.std(M_param_stdv)

S_mean_stdv = np.std(S_param_mean)
S_stdv_stdv = np.std(S_param_stdv)

# Which noise level and fit to the sample do you wish to show?
show_noise = 99

mp.plot(range1, G_noise[show_noise], "blue", linewidth = 2)
mp.plot(range1, gaussian(range1, G_param_mean[show_noise], G_param_stdv[show_noise]), "red", linewidth = 2)
mp.title("Random sample of values with added noise and fit (Gaussian)")
mp.show()

mp.plot(range1, M_noise[show_noise], "blue", linewidth = 2)
mp.plot(range1, moffat(range1, M_param_mean[show_noise], M_param_stdv[show_noise]), "red", linewidth = 2)
mp.title("Random sample of values with added noise and fit (Moffat)")
mp.show()

mp.plot(range1, S_noise[show_noise], "blue", linewidth = 2)
mp.plot(range1, sinc(range1, S_param_mean[show_noise], S_param_stdv[show_noise]), "red", linewidth = 2)
mp.title("Random sample of values with added noise and fit (Sinc)")
mp.show()


mp.hist(G_param_stdv, normed = 100)



