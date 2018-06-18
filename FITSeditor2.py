# -*- coding: utf-8 -*-
"""

@author: Marcus Karlsson
"""

from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
from astropy.modeling import models, fitting

fitsimage_1 = fits.open('bpic_0_2.fits')

fitsimage_1.info()

fits1 = fitsimage_1[0]

#print(fits1.data.shape)

#print(fits1.header)

# Extracts all the pixel data from the image
#Image_Data_All2 is the one used for the fit, it includes 100 rows of pixels centered
#on the brightest line in the image
Image_Data_0 = fits1.data[:,0]
Image_Data2_0 = Image_Data_0[300:400]
Image_Data_All = fits1.data[:,:]
Image_Data_All2 = Image_Data_All[300:400]

#Define a running median calculator
def RunMedian(x,N):
    idx = np.arange(N) + np.arange(len(x)-N+1)[:,None]
    b = [row[row>0] for row in x[idx]]
    return np.array(map(np.median,b))


#Create the Gaussian and Moffat fits and creates Fit_Data and Fit_Data2
#which contains the parameters for the two models (Moffat still does not work correct)
x = np.linspace(-50,50,100)
Gauss_Model = models.Gaussian1D(amplitude = 1000., mean = 0, stddev = 1.)
Moffat_Model = models.Moffat1D(amplitude = 1000, x_0 = 0, gamma = 1, alpha = 2)
Fitting_Model = fitting.LevMarLSQFitter()

Fit_Data = []
Fit_Data_2 = []

for i in range(0, Image_Data_All2.shape[1]):
    Fit_Data.append(Fitting_Model(Gauss_Model, x, Image_Data_All2[:,i]))

for i in range(0, Image_Data_All2.shape[1]):
    Fit_Data_2.append(Fitting_Model(Moffat_Model, x, Image_Data_All2[:,i]))


print(Fit_Data[0])
print(Fit_Data[0].amplitude)
print(Fit_Data_2[0])
print(Fit_Data_2[0].amplitude)

plt.plot(x, Image_Data_All2[:,0])
plt.plot(x, Fit_Data[0](x))
plt.axis((-12, 12, 0, 800))
#plt.plot(x, Image_Data_All2[:,1023])
#plt.plot(x, Fit_Data[1023](x))
plt.show()

plt.plot(x, Image_Data_All2[:,0])
plt.plot(x, Fit_Data_2[0](x))
#plt.plot(x, Image_Data_All2[:,1023])
#plt.plot(x, Fit_Data_2[1023](x))
plt.show()


#Extracts the individual parameters for the Gaussian fits of each column into three
# arrays of the amplitude, mean, and stdv
Fit_Data[0].parameters

x2 = np.empty((1024,3))

for n in range(1024):
    x2[n,:] = Fit_Data[n].parameters

Amp_Data = x2[:,0]
Mean_Data = x2[:,1]
Stdv_Data = x2[:,2]

#plt.plot(Amp_Data)
#plt.show()

#plt.plot(Mean_Data)
#plt.show()

#plt.plot(Stdv_Data)
#plt.show()



#Creates a residual array of the difference of value between the actual pixel value and
#the values created by the Gaussian fit
FitGaussian = np.empty((100,1024))

for n in range(1024):
    FitGaussian[:,n] = Fit_Data[n](x)

Residual = Image_Data_All2 - FitGaussian

#Plots the residual for the first column
#plt.plot(x, Residual[:,0], 'ro')
#plt.plot(x, Residual[:,1023])
#plt.show()

#Image_Amp = []
#Image_Mean = []
#Image_Stddev = []

#for n in range(1024):
#    Image_Amp.append(max(Image_Data_All[:,n]))


#Residual_Amp = Amp_Data - Image_Amp

#plt.plot(Residual_Amp, '.r')
#plt.axis([0, 1024, -50, 420])
#plt.show()



