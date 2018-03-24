# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 14:27:52 2018

@author: Marcus Karlsson
"""

#Import necessary functions and libraries
from matplotlib import pyplot as mp
from matplotlib import mlab as mlab
import numpy as np
import math as math
import pylab as pylab
import scipy as sp
from scipy.stats import norm
from astropy.modeling.models import Moffat1D
from scipy.optimize import curve_fit


#Define constants (mean, FWHM, sigma, noise)
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


# Define line profiles of Gaussian, Sinc, Moffat distributions
def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2*(sigma)**2))

# 1/(np.sqrt(2 * math.pi * sigma ** 2))*

def sinc(x):
    return pylab.sin(x) / x

def moffat(x, mu3, sigma3):
    return (((x - mu3) / (sigma3)) ** 2.0 + 1.0) ** (-beta)


# Plot distributions
#mp.plot(range1, gaussian(range1, mu, sigma)+noise)
#mp.show()

#mp.plot(range, sinc(range)+noise)
#mp.show()

#mp.plot(range, moffat(range, mu3, sigma3)+noise)
#mp.show()


# Fit data to a Gaussian distribution
G_sample_test1 = gaussian(range1, 1, 0, 1)
# [np.random.normal(mu, sigma, points)]
G_sample_test2 = []
for i in range(points):
    G_sample_test2.append(G_sample_test1)

G_noise_test = [G_sample_test2[i] + noise3[i] for i in range(points)]

G_initial = [1.0, 0.0, 1.0]

G_param_test1 = [curve_fit(gaussian, range1, G_noise_test[i], p0 = G_initial) for i in range(points)]
G_param, G_covar = map(list, zip(*G_param_test1))
#G_param_test2 = [norm.fit(G_noise_test[i]) for i in range(points)]

G_param_mean = [i for _,i,_ in G_param]

G_param_stdv = [i for _,_,i in G_param]


#G_fit_pdf = [gaussian(range1, G_param_mean[i], G_param_stdv[i]) for i in range(points)]
M_diff = abs(np.linspace(0, 0, points) - G_param_mean)
S_diff = abs(np.linspace(1, 1, points) - G_param_stdv)

#mp.plot(range1, G_fit_pdf[0], "red", label = "Fitted gaussian dist.", linestyle = "dashed", linewidth = 2)
#mp.plot(range1, G_noise_test[99], color = "green")
#mp.title("Gaussian distribution fitting")
#mp.legend()
#mp.show()

M_line = np.polyfit(range2, M_diff, 1)
S_line = np.polyfit(range2, S_diff, 1)

mp.plot(range2, M_diff, "red", linewidth = 2)
mp.plot(range2, M_line[1] + range2 * M_line[0], "blue", linewidth = 2)
mp.title("Difference in mean with increasing noise")
mp.xlabel("Maximum amplitude of noise")
mp.ylabel("Distance from actual mean = 0")
mp.show()

mp.plot(range2, S_diff, "blue", linewidth = 2)
mp.plot(range2, S_line[1] + range2 * S_line[0], "red", linewidth = 2)
mp.title("Difference in standard deviation with increasing noise")
mp.xlabel("Maximum amplitude of noise")
mp.ylabel("Distance from actual stdv = 1")
mp.show()










# Generate an list of 100 equal arrays of 100 random values from a
# Gaussian distribution with mean = 0 and stdv = 1
Gaussian_sample1 = [norm.rvs(loc = 0, scale = 1, size = points)]
Gaussian_sample2 = []
for i in range(points):
    Gaussian_sample2.append(Gaussian_sample1[0])

# Add the noise levels to these 100 arrays 
# with increasing amplitudes
Gaussian_sample_noise = [Gaussian_sample2[i] + noise3[i] for i in range(points)]

# Generate a Gaussian fit to the sample values
# and returns a list of two parameters
# (mean, stdv) at Gaussian_parameters[0, 1]
# based on MLE of the given data.
Gaussian_parameters = [norm.fit(Gaussian_sample_noise[i]) for i in range(points)]
G_parameter_mean = [i for i,_ in Gaussian_parameters]
G_parameter_stdv = [i for _,i in Gaussian_parameters]

# Generate the fitted distribution 
# (probability density function)
# over range1 with parameters[0, 1]
fitted_pdf = [norm.pdf(range1,loc = G_parameter_mean[i], scale = G_parameter_stdv[i]) for i in range(points)]

# Generate the Gaussian distribution non-fitted
# (probability density function)
# over range1 with parameters[0, 1]
normal_pdf = norm.pdf(range1)

#mp.plot(range1, fitted_pdf[0], "red", label = "Fitted gaussian dist.", linestyle = "dashed", linewidth = 2)
#mp.plot(range1, normal_pdf, "blue", label = "Gaussian dist.", linewidth = 2)
#mp.hist(Gaussian_sample2[0], normed = 1, color = "green", alpha = .3)
#mp.title("Gaussian distribution fitting")
#mp.legend()
#mp.show()

# Test how well the fitted distribution coincides
# with the data of the regular Gaussian
Mean_difference = abs(np.linspace(0, 0, points) - G_parameter_mean)
Stdv_difference = abs(np.linspace(1, 1, points) - G_parameter_stdv)

# Plot the difference in the given mean with the
# fitted mean as a function of increasing noise
#mp.plot(range2, Mean_difference, "red", linewidth = 2)
#mp.title("Difference in mean with increasing noise")
#mp.xlabel("Maximum amplitude of noise")
#mp.ylabel("Distance from actual mean = 0")
#mp.show()


# Plot the difference in the given stdv with the
# fitted stdv as a function of increasing noise
#mp.plot(range2, Stdv_difference, "blue", linewidth = 2)
#mp.title("Difference in standard deviation with increasing noise")
#mp.xlabel("Maximum amplitude of noise")
#mp.ylabel("Distance from actual stdv = 1")
#mp.show()


