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
mu = 0
fwhm = 5
sigma = fwhm / (2.0*math.sqrt(2.0 * math.log(2.0)))
mu3 = 0
fwhm3 = 5
beta = 1
sigma3 = fwhm3 / (2.0 * math.sqrt(2.0 ** (1.0 / beta) - 1.0))

# Define how many data points you wish to work with
points = 100

# Define how large you wish the noise amplitude to be
noise_amp = 0.2


range1 = np.linspace(-5, 5, points)
# Creates a random noise array with input
# (Lowest value, Highest value, Number of elements)
noise = abs(np.random.uniform(0, noise_amp, points))

# Creates a range of values used as amplitude 
# for different noise levels with input
# (Starting value, End value, Number of Steps)
range2 = np.linspace(0, noise_amp, points)

# Creates a list of noise arrays with input
# (Lowest value, Highest value, Number of elements)
# Where the input of x in range(x) determines
# the number of arrays
noise1 = [np.random.normal(0, 1, points) for i in range(points)]

# Normalize the noise to range -1:1 if needed
noise2 = [2*((noise1[i]-min(noise1[i]))/(max(noise1[i])-min(noise1[i]))) - 1 for i in range(points)]

# Multiplies the list of noise arrays with the range
# of numbers defined by range2, such that each
# noise array has a maximum amplitude of each step
# in the range in range2
noise3 = [noise2[i] * range2[i] for i in range(points)]


# Fit data to a Gaussian distribution

def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2*(sigma)**2))

# Generate an list of 100 equal arrays of 100 random values from a
# Gaussian distribution with amp = 1, mean = 0, and stdv = 1
G_sample_test1 = gaussian(range1, 1, 0, 1)

G_sample_test2 = []
for i in range(points):
    G_sample_test2.append(G_sample_test1)

# Add the generated noise at 100 different amplitude levels to
# the 100 arrays of gaussian values 'G_sample_test2'

G_noise_test = [G_sample_test2[i] + noise3[i] for i in range(points)]

# Define the initial values of the amp, mean, and stdv the fit should
# guess the fitted values from
G_initial = [1.0, 0.0, 1.0]

# Create a fit to the gaussian parameters using 'curve_fit' which generates a
# tuple array consisting of a fit to [amp, mean, stdv] and the covariance
G_param_test1 = [curve_fit(gaussian, range1, G_noise_test[i], p0 = G_initial) for i in range(points)]

# Splitting up the parameters and the covariance into two arrays
G_param, G_covar = map(list, zip(*G_param_test1))

# Extracting the fit to the mean parameters of the 100 gaussian arrays
G_param_mean = [i for _,i,_ in G_param]

# Extracting the fit to the stdv parameters of the 100 gaussian arrays 
G_param_stdv = [i for _,_,i in G_param]

# Creates two arrays to see how much the fit to the mean and the stdv
# differentiates from the actual mean and stdv of the gaussian values
# Mean = 0, Standard deviation = 1
M_diff = abs(np.linspace(0, 0, points) - G_param_mean)
S_diff = abs(np.linspace(1, 1, points) - G_param_stdv)

# Creates a linear fit to how the difference in fitted and actual
# mean and stdv changes with different amplitude of noise
M_line = np.polyfit(range2, M_diff, 1)
S_line = np.polyfit(range2, S_diff, 1)

# Plots the variation in fitted mean from actual mean = 0 as function
# of increasing noise amplitude and a linear fit to the values
mp.plot(range2, M_diff, "red", linewidth = 2)
mp.plot(range2, M_line[1] + range2 * M_line[0], "blue", linewidth = 2)
mp.title("Difference in mean with increasing noise (Gaussian)")
mp.xlabel("Maximum amplitude of noise")
mp.ylabel("Distance from actual mean = 0")
mp.show()

# Plots the variation in fitted stdv from actual stdv = 1 as function
# of increasing noise amplitude and a linear fit to the values
mp.plot(range2, S_diff, "blue", linewidth = 2)
mp.plot(range2, S_line[1] + range2 * S_line[0], "red", linewidth = 2)
mp.title("Difference in standard deviation with increasing noise (Gaussian)")
mp.xlabel("Maximum amplitude of noise")
mp.ylabel("Distance from actual stdv = 1")
mp.show()

#G_fit_pdf = [gaussian(range1, G_param_mean[i], G_param_stdv[i]) for i in range(points)]
#mp.plot(range1, G_fit_pdf[0], "red", label = "Fitted gaussian dist.", linestyle = "dashed", linewidth = 2)
#mp.plot(range1, G_noise_test[99], color = "green")
#mp.title("Gaussian distribution fitting")
#mp.legend()
#mp.show()


# Fit data to a Moffat distribution

def moffat(x, a2, mu2, sigma2, beta):
    return a2 * (((x - mu2) / (sigma2)) ** 2.0 + 1.0) ** (-beta)

# Generate an list of 100 equal arrays of 100 random values from a
# Moffat distribution with amp = 1, mean = 0, and stdv = 1
M_sample_test1 = moffat(range1, 1, 0, 1, 1)

M_sample_test2 = []
for i in range(points):
    M_sample_test2.append(M_sample_test1)

# Add the generated noise at 100 different amplitude levels to
# the 100 arrays of moffat values 'M_sample_test2'
M_noise_test = [M_sample_test2[i] + noise3[i] for i in range(points)]

# Define the initial values of the amp, mean, and stdv the fit should
# guess the fitted values from
M_initial = [1.0, 0.0, 1.0, 1.0]

# Create a fit to the Moffat parameters using 'curve_fit' which generates a
# tuple array consisting of a fit to [amp, mean, stdv] and the covariance
M_param_test1 = [curve_fit(moffat, range1, M_noise_test[i], p0 = M_initial) for i in range(points)]

# Splitting up the parameters and the covariance into two arrays
M_param, M_covar = map(list, zip(*M_param_test1))

# Extracting the fit to the mean parameters of the 100 Moffat arrays
M_param_mean = [i for _,i,_,_ in M_param]

# Extracting the fit to the stdv parameters of the 100 Moffat arrays 
M_param_stdv = [i for _,_,i,_ in M_param]

# Creates two arrays to see how much the fit to the mean and the stdv
# differentiates from the actual mean and stdv of the moffat values
# Mean = 0, Standard deviation = 1
M_diff_moffat = abs(np.linspace(0, 0, points) - M_param_mean)
S_diff_moffat = abs(np.linspace(1, 1, points) - M_param_stdv)

# Creates a linear fit to how the difference in fitted and actual
# mean and stdv changes with different amplitude of noise
M_line_moffat = np.polyfit(range2, M_diff_moffat, 1)
S_line_moffat = np.polyfit(range2, S_diff_moffat, 1)

# Plots the variation in fitted mean from actual mean = 0 as function
# of increasing noise amplitude and a linear fit to the values
mp.plot(range2, M_diff_moffat, "red", linewidth = 2)
mp.plot(range2, M_line_moffat[1] + range2 * M_line_moffat[0], "blue", linewidth = 2)
mp.title("Difference in mean with increasing noise (Moffat)")
mp.xlabel("Maximum amplitude of noise")
mp.ylabel("Distance from actual mean = 0")
mp.show()

# Plots the variation in fitted stdv from actual stdv = 1 as function
# of increasing noise amplitude and a linear fit to the values
mp.plot(range2, S_diff_moffat, "blue", linewidth = 2)
mp.plot(range2, S_line_moffat[1] + range2 * S_line_moffat[0], "red", linewidth = 2)
mp.title("Difference in standard deviation with increasing noise (Moffat)")
mp.xlabel("Maximum amplitude of noise")
mp.ylabel("Distance from actual stdv = 1")
mp.show()


# Fit data to a Sinc distribution

def sinc(x, a3, mu3, sigma3):
    return pylab.sin(math.pi * (x - mu3) * sigma3) / (math.pi * (x - mu3) * sigma3)

# Generate an list of 100 equal arrays of 100 random values from a
# Sinc distribution with amp = 1, mean = 0, and stdv = 1
S_sample_test1 = sinc(range1, 1, 0, 1)

S_sample_test2 = []
for i in range(points):
    S_sample_test2.append(S_sample_test1)

# Add the generated noise at 100 different amplitude levels to
# the 100 arrays of sinc values 'S_sample_test2'
S_noise_test = [S_sample_test2[i] + noise3[i] for i in range(points)]

# Define the initial values of the amp, mean, and stdv the fit should
# guess the fitted values from
S_initial = [1.0, 0.0, 1.0]

# Create a fit to the sinc parameters using 'curve_fit' which generates a
# tuple array consisting of a fit to [amp, mean, stdv] and the covariance
S_param_test1 = [curve_fit(sinc, range1, S_noise_test[i], p0 = S_initial) for i in range(points)]

# Splitting up the parameters and the covariance into two arrays
S_param, S_covar = map(list, zip(*S_param_test1))

# Extracting the fit to the mean parameters of the 100 sinc arrays
S_param_mean = [i for _,i,_,_ in M_param]

# Extracting the fit to the stdv parameters of the 100 sinc arrays 
S_param_stdv = [i for _,_,i,_ in M_param]

# Creates two arrays to see how much the fit to the mean and the stdv
# differentiates from the actual mean and stdv of the sinc values
# Mean = 0, Standard deviation = 1
M_diff_sinc = abs(np.linspace(0, 0, points) - S_param_mean)
S_diff_sinc = abs(np.linspace(1, 1, points) - S_param_stdv)

# Creates a linear fit to how the difference in fitted and actual
# mean and stdv changes with different amplitude of noise
M_line_sinc = np.polyfit(range2, M_diff_sinc, 1)
S_line_sinc = np.polyfit(range2, S_diff_sinc, 1)

# Plots the variation in fitted mean from actual mean = 0 as function
# of increasing noise amplitude and a linear fit to the values
mp.plot(range2, M_diff_sinc, "red", linewidth = 2)
mp.plot(range2, M_line_sinc[1] + range2 * M_line_sinc[0], "blue", linewidth = 2)
mp.title("Difference in mean with increasing noise (Sinc)")
mp.xlabel("Maximum amplitude of noise")
mp.ylabel("Distance from actual mean = 0")
mp.show()

# Plots the variation in fitted stdv from actual stdv = 1 as function
# of increasing noise amplitude and a linear fit to the values
mp.plot(range2, S_diff_sinc, "blue", linewidth = 2)
mp.plot(range2, S_line_sinc[1] + range2 * S_line_sinc[0], "red", linewidth = 2)
mp.title("Difference in standard deviation with increasing noise (Sinc)")
mp.xlabel("Maximum amplitude of noise")
mp.ylabel("Distance from actual stdv = 1")
mp.show()


# Parameter fit standard deviations:

G_mean_stdv = np.std(G_param_mean)
G_stdv_stdv = np.std(G_param_stdv)

M_mean_stdv = np.std(M_param_mean)
M_stdv_stdv = np.std(M_param_stdv)

S_mean_stdv = np.std(S_param_mean)
S_stdv_stdv = np.std(S_param_stdv)

mp.plot()











