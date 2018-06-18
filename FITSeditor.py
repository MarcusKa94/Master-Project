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

Image_Data_0 = fits1.data[:,0]
Image_Data2_0 = Image_Data_0[325:375]
Image_Data_All = fits1.data[:,:]
Image_Data_All2 = Image_Data_All[325:375]
x = np.linspace(-50,50,50)

Gauss_Model = models.Gaussian1D(amplitude = 1000., mean = 0, stddev = 1.)
Moffat_Model = models.Moffat1D(amplitude = 1000., x_0 = 0, gamma = 2., alpha = 1.)


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
plt.plot(x, Image_Data_All2[:,1023])
plt.plot(x, Fit_Data[1023](x))
plt.show()

plt.plot(x, Image_Data_All2[:,0])
plt.plot(x, Fit_Data_2[0](x))
plt.plot(x, Image_Data_All2[:,1023])
plt.plot(x, Fit_Data_2[1023](x))
plt.show()





Fit_Data[0].parameters

x2 = np.empty((1024,3))

for n in range(1024):
    x2[n,:] = Fit_Data[n].parameters

Amp_Data = x2[:,0]
Mean_Data = x2[:,1]
Stdv_Data = x2[:,2]

plt.plot(Amp_Data)
plt.show()

plt.plot(Mean_Data)
plt.show()

plt.plot(Stdv_Data)
plt.show()

Image_Amp = []
Image_Mean = []
Image_Stddev = []

for n in range(1024):
    Image_Amp.append(max(Image_Data_All[:,n]))

for n in range(1024):
    Image_Mean = (sum(Image_Data_All2[:,n]) / len(Image_Data_All2[:,n]))




Residual_Amp = Amp_Data - Image_Amp

plt.plot(Residual_Amp, '.r')
#plt.axis([0, 1024, -50, 420])
plt.show()




