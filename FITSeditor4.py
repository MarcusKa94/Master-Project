# -*- coding: utf-8 -*-
"""

@author: Marcus Karlsson
"""

from astropy.io import fits
from astropy.stats import sigma_clip
import numpy as np
from matplotlib import pyplot as plt
from astropy.modeling import models, fitting
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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
Image_Data_All2= np.nan_to_num(Image_Data_All2)

#Define a running median calculator
def RunMedian(x,N):
    idx = np.arange(N) + np.arange(len(x)-N+1)[:,None]
    b = [row[row>0] for row in x[idx]]
    return np.array(map(np.median,b))


#Create the Gaussian and Moffat fits and creates Fit_Data and Fit_Data2
#which contains the parameters for the two models (Moffat still does not work correct)
x = np.linspace(-50,50,100)
Gauss_Model = models.Gaussian1D(amplitude = 1000., mean = 0, stddev = 1., )
Gauss_Model.mean.fixed = False
Moffat_Model = models.Moffat1D(amplitude = 1000, x_0 = 0, gamma = 1, alpha = 2)
Fitting_Model = fitting.LevMarLSQFitter()

Fit_Data_1 = []
Fit_Data_2 = []


for i in range(0, Image_Data_All2.shape[1]):
    Fit_Data_1.append(Fitting_Model(Gauss_Model, x, Image_Data_All2[:,i]))

x2 = np.empty((1024,3))

for n in range(1024):
    x2[n,:] = Fit_Data_1[n].parameters

Amp_Data_1 = x2[:,0]
Mean_Data_1 = x2[:,1]
Stdv_Data_1 = x2[:,2]

for i in range(0, Image_Data_All2.shape[1]):
    if Fit_Data_2:  # true if not an empty list
        Gauss_Model_2 = models.Gaussian1D(amplitude=Fit_Data_2[-1].amplitude,
                                        mean=Mean_Data_1[i],
                                        stddev=Stdv_Data_1[i])
        Gauss_Model_2.mean.fixed = True
        Gauss_Model_2.stddev.fixed = True
    Fit_Data_2.append(Fitting_Model(Gauss_Model_2, x, Image_Data_All2[:,i]))

x3 = np.empty((1024,3))

for n in range(1024):
    x3[n,:] = Fit_Data_2[n].parameters

Amp_Data_2 = x3[:,0]
Mean_Data_2 = x3[:,1]
Stdv_Data_2 = x3[:,2]


#for i in range(0, Image_Data_All2.shape[1]):
    #Fit_Data_4.append(Fitting_Model(Moffat_Model, x, Image_Data_All2[:,i]))

Moffat_Model = models.Moffat1D(amplitude = 1000, x_0 = 0, gamma = 1, alpha = 2)
Fit_Data_4 = []
Lorentz_Model = models.Lorentz1D(amplitude = 1000, x_0 = 0, fwhm = 1)
Lorentz_Model.x_0.fixed = False

for i in range(0, Image_Data_All2.shape[1]):
    Fit_Data_4.append(Fitting_Model(Lorentz_Model, x, Image_Data_All2[:,i]))

x5 = np.empty((1024,3))
for n in range(1024):
    x5[n,:] = Fit_Data_4[n].parameters

Amp_Data_4 = x5[:,0]
Mean_Data_4 = x5[:,1]
FWHM_Data_4 = x5[:,2]


plt.plot(x, Image_Data_All2[:,0])
plt.plot(x, Fit_Data_4[0](x))
#plt.plot(x, Image_Data_All2[:,1023])
#plt.plot(x, Fit_Data_1[1023](x))
plt.show()

FitLorentz_1 = np.empty((100,1024))

for n in range(1024):
    FitLorentz_1[:,n] = Fit_Data_4[n](x)

Residual_4 = Image_Data_All2 - FitLorentz_1

#Plots the residual for the first column
plt.plot(x, Residual_4[:,0], 'ro')
#plt.plot(x, Residual[:,1023])
plt.show()

#Polynomial fitting test to Mean_Data_2
Polydegree = 3
Mean_Range = np.linspace(0, 1023, 1024)

Polytest = np.polyfit(Mean_Range, Mean_Data_2, Polydegree)
Polyfit_Function = Polytest[0]*Mean_Range**3 + Polytest[1]*Mean_Range**2 + Polytest[2]*Mean_Range**1 + Polytest[3]
plt.plot(Mean_Data_2)
plt.plot(Polyfit_Function)
plt.show()


Mean_Residual = (Mean_Data_2 - Polyfit_Function)
Mean_Residual = sigma_clip(Mean_Residual, sigma = 5)
Mean_Residual = sigma_clip(Mean_Residual, sigma = 3)

plt.plot(Mean_Residual)
plt.show()

Mean_Data_Clipped = Mean_Residual + Polyfit_Function
Mean_Data_Clipped = Mean_Data_Clipped[~Mean_Data_Clipped.mask]
plt.plot(Mean_Data_Clipped)
plt.show()

Mean_Range_2 = np.linspace(0, Mean_Data_Clipped.shape[0]-1, Mean_Data_Clipped.shape[0])

Polytest_2 = np.polyfit(Mean_Range_2, Mean_Data_Clipped, Polydegree)
Polyfit_Function_2 = Polytest_2[0]*Mean_Range_2**3 + Polytest_2[1]*Mean_Range_2**2 + Polytest_2[2]*Mean_Range_2**1 + Polytest_2[3]
plt.plot(Mean_Data_Clipped)
plt.plot(Polyfit_Function_2)
plt.show()


Mean_Residual_2 = (Mean_Data_Clipped - Polyfit_Function_2)
Mean_Residual_2 = sigma_clip(Mean_Residual_2, sigma = 5)
Mean_Residual_2 = sigma_clip(Mean_Residual_2, sigma = 3)

plt.plot(Mean_Residual_2)
plt.show()




















Fit_Image = []

for i in range(0, Image_Data_All2.shape[1]):
    Fit_Image.append(Fit_Data_1[i](x))

Fitted_Image = np.array(Fit_Image)

plt.pcolormesh(np.transpose(Fitted_Image))
plt.show()

plt.pcolormesh(Image_Data_All2)
plt.show

Image_Residual = Image_Data_All2 - np.transpose(Fitted_Image)

plt.pcolormesh(Image_Residual, cmap='RdBu')
plt.colorbar()
plt.show()



ax_x = np.linspace(1, 100, 100) #range(Image_Residual.shape[0])
ax_y = np.linspace(1, 1024, 1024) #range(Image_Residual.shape[1])
aX, aY = np.meshgrid(ax_x, ax_y)

fig = plt.figure(figsize = (9, 6))
#ax = fig.gca(projection = '3d')
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(-150.0, 150.0)
Image_Residual[13, 210] = 0
Residual_3D = ax.plot_surface(aX, aY, np.transpose(Image_Residual), cmap=cm.coolwarm)
fig.colorbar(Residual_3D, shrink=0.6, aspect=5)
plt.show()


print(Fit_Data_1[1])
print(Fit_Data_1[1].amplitude)
print(Fit_Data_2[1])
print(Fit_Data_2[1].amplitude)


plt.plot(x, Image_Data_All2[:,0])
Gaussiantest = models.Gaussian1D(amplitude = 735., mean = 2.47133, stddev = 1.5)
plt.plot(x, Gaussiantest(x))
plt.axis([2.4, 2.6, 730, 735])
#plt.axis([-15, 15, -10, 200])
plt.show()

plt.plot(x, Fit_Data_2[0](x))
plt.plot(x, Image_Data_All2[:,1023])
plt.plot(x, Fit_Data_2[1023](x))
plt.show()

plt.plot(x, Image_Data_All2[:,0])
plt.plot(x, Fit_Data_4[0](x))
#plt.plot(x, Image_Data_All2[:,1023])
#plt.plot(x, Fit_Data_1[1023](x))
plt.show()


#Extracts the individual parameters for the Gaussian fits of each column into three
# arrays of the amplitude, mean, and stdv
Fit_Data_2[0].parameters

#plt.plot(Amp_Data)
#plt.show()

#plt.plot(Mean_Data)
#plt.show()

#plt.plot(Stdv_Data)
#plt.show()


#Creates a residual array of the difference of value between the actual pixel value and
#the values created by the Gaussian fit
FitGaussian_1 = np.empty((100,1024))
FitGaussian_2 = np.empty((100,1024))

for n in range(1024):
    FitGaussian_1[:,n] = Fit_Data_1[n](x)

Residual_1 = Image_Data_All2 - FitGaussian_1

for n in range(1024):
    FitGaussian_2[:,n] = Fit_Data_2[n](x)

Residual_2 = Image_Data_All2 - FitGaussian_2


#Plots the residual for the first column
plt.plot(x, Residual_1[:,0], 'ro')
#plt.plot(x, Residual[:,1023])
plt.show()

plt.plot(x, Residual_2[:,0], 'ro')
#plt.plot(x, Residual_2[:,1023])
plt.show()

#Image_Amp = []
#Image_Mean = []
#Image_Stddev = []

#for n in range(1024):
#    Image_Amp.append(max(Image_Data_All[:,n]))


#Residual_Amp = Amp_Data - Image_Amp

#plt.plot(Residual_Amp, '.r')
#plt.axis([0, 1024, -50, 420])
#plt.show()

from lmfit.models import GaussianModel

x = np.linspace(-50,50,100)

# create a model for a Gaussian
Gauss_Model_3 = GaussianModel()

# make a set of parameters, setting initial values
params = Gauss_Model_3.make_params(amplitude=1000, center=0, sigma=1.0)

Fit_Results = []

for i in range(Image_Data_All2.shape[1]):
    result = Gauss_Model_3.fit(Image_Data_All2[:, i], params, x=x, nan_policy='omit')
    Fit_Results.append(result)
    # update `params` with the current best fit params for the next column
    params = result.params


#plt.plot(x, Image_Data_All2[:,0])
#plt.plot(x, Fit_Results[0](x))
#plt.show()

x4 = np.empty((1024,3))

for n in range(1024):
    x4[n,:] = Fit_Data_2[n].parameters

Amp_Data_3 = x4[:,0]
Mean_Data_3 = x4[:,1]
Stdv_Data_3 = x4[:,2]





