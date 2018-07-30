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
import os

os.chdir('C:/Users/marcu/Documents/Mastersarbete 2018')

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

#Normal Gauss Model Fitting

for i in range(0, Image_Data_All2.shape[1]):
    Fit_Data_1.append(Fitting_Model(Gauss_Model, x, Image_Data_All2[:,i]))

x2 = np.empty((1024,3))

for n in range(1024):
    x2[n,:] = Fit_Data_1[n].parameters

Amp_Data_1 = x2[:,0]
Mean_Data_1 = x2[:,1]
Stdv_Data_1 = x2[:,2]

#Running Median Gauss Model Fitting

Fit_Data_2 = []
Gauss_Model_2 = []
for i in range(0, Image_Data_All2.shape[1]):
    if Fit_Data_1:  # true if not an empty list
        Gauss_Model_2 = models.Gaussian1D(amplitude=Amp_Data_1[-1],
                                        mean=Mean_Data_1[i],
                                        stddev=Stdv_Data_1[i])
        Gauss_Model_2.mean.fixed = True
        Gauss_Model_2.stddev.fixed = False
    Fit_Data_2.append(Fitting_Model(Gauss_Model_2, x, Image_Data_All2[:,i]))

x3 = np.empty((1024,3))

for n in range(1024):
    x3[n,:] = Fit_Data_2[n].parameters

Amp_Data_2 = x3[:,0]
Mean_Data_2 = x3[:,1]
Stdv_Data_2 = x3[:,2]



#Polynomial fitting test to Mean_Data_2
Mean_Data_2 = x3[:,1]
Mean_Data_Clipped = np.zeros((1023,))
Mean_Data_Residual = np.zeros((1024,))

while len(Mean_Data_Clipped) < len(Mean_Data_Residual):
    if len(Mean_Data_Clipped) < len(Mean_Data_Residual):
        Polydegree = 2
        Mean_Range = np.linspace(0, len(Mean_Data_2)-1, len(Mean_Data_2))
        Polytest = np.polyfit(Mean_Range, Mean_Data_2, Polydegree)
        Polyfit_Function = Polytest[0]*Mean_Range**2 + Polytest[1]*Mean_Range**1 + Polytest[2]
        Mean_Data_Residual = (Mean_Data_2 - Polyfit_Function)
        Mean_Data_Residual = sigma_clip(Mean_Data_Residual, sigma = 5)
        Mean_Data_Residual = sigma_clip(Mean_Data_Residual, sigma = 3)
        Mean_Data_Clipped = Mean_Data_Residual + Polyfit_Function
        #Mean_Data_Clipped = Mean_Data_Clipped[~Mean_Data_Clipped.mask]
        Mean_Data_2 = Mean_Data_Clipped
    else:
        print ('Sigma Clipping Complete')

Mean_Poly_Fit = np.ma.polyfit(Mean_Range, Mean_Data_Clipped, Polydegree)
Mean_Range_Fit = np.linspace(0, Image_Data_All2.shape[1]-1, Image_Data_All2.shape[1])
Mean_Data_Fit = Mean_Poly_Fit[0]*Mean_Range_Fit**2 + Mean_Poly_Fit[1]*Mean_Range_Fit**1 + Mean_Poly_Fit[2]

plt.plot(Mean_Data_2)
plt.plot(Mean_Data_Fit)
plt.show()



##Normal Lorentz Model Fitting

Fit_Data_4 = []
Lorentz_Model = models.Lorentz1D(amplitude = 1000, x_0 = 0, fwhm = 1)
Lorentz_Model.x_0.fixed = False

#for i in range(0, Image_Data_All2.shape[1]):
#    Fit_Data_4.append(Fitting_Model(Lorentz_Model, x, Image_Data_All2[:,i]))

for i in range(0, Image_Data_All2.shape[1]):
    if Fit_Data_4:  # true if not an empty list
        #Amp_Data_4[1023] = Amp_Data_4[0]
        Lorentz_Model = models.Lorentz1D(amplitude=Amp_Data_2[-1],
                                        #x_0 = Mean_Data_4[i],
                                        x_0 = Mean_Data_Fit[i],
                                        fwhm = 1)
        Lorentz_Model.x_0.fixed = True
    Fit_Data_4.append(Fitting_Model(Lorentz_Model, x, Image_Data_All2[:,i]))


x5 = np.empty((1024,3))
for n in range(1024):
    x5[n,:] = Fit_Data_4[n].parameters

Amp_Data_4 = x5[:,0]
Mean_Data_4 = x5[:,1]
FWHM_Data_4 = x5[:,2]


plt.plot(x, Image_Data_All2[:,1])
plt.plot(x, Fit_Data_4[1](x))
#plt.plot(x, Image_Data_All2[:,1023])
#plt.plot(x, Fit_Data_1[1023](x))
plt.show()

FitLorentz_1 = np.empty((100,1024))

for n in range(1024):
    FitLorentz_1[:,n] = Fit_Data_4[n](x)

Residual_4 = Image_Data_All2 - FitLorentz_1

#Plots the residual for the first column
plt.plot(x, Residual_4[:,1], 'ro')
#plt.plot(x, Residual[:,1023])
plt.show()



#Normal Moffat Model Fitting

Moffat_Model = models.Moffat1D(amplitude = 1000, x_0 = 0, gamma = 1, alpha = 2)

Fit_Data_3 = []

#for i in range(0, Image_Data_All2.shape[1]):
#    Fit_Data_3.append(Fitting_Model(Moffat_Model, x, Image_Data_All2[:,i]))

for i in range(0, Image_Data_All2.shape[1]):
    if Fit_Data_3:  # true if not an empty list
        #Amp_Data_4[1023] = Amp_Data_4[0]
        Moffat_Model = models.Moffat1D(amplitude=Amp_Data_4[-1],
                                        #x_0 = Mean_Data_4[i],
                                        x_0 = Mean_Data_Fit[i],
                                        gamma = 1.3, alpha = 0.9)
        Moffat_Model.x_0.fixed = True
    Fit_Data_3.append(Fitting_Model(Moffat_Model, x, Image_Data_All2[:,i]))
    Fit_Data_3[0] = Fitting_Model(models.Moffat1D(amplitude=Amp_Data_4[0],
                                        x_0 = Mean_Data_Fit[0],
                                        gamma = 1.3, alpha = 0.9),
                                        x, Image_Data_All2[:,0])

x6 = np.empty((1024,4))
for n in range(1024):
    x6[n,:] = Fit_Data_3[n].parameters

Amp_Data_5 = x6[:,0]
Mean_Data_5 = x6[:,1]
Gamma_Data_5 = x6[:,2]
Alpha_Data_5 = x6[:,3]


#Redo Moffat fit to fix the gamma parameter which is not fitted in a single loop

Fit_Data_3 = []

for i in range(0, Image_Data_All2.shape[1]):
    if Fit_Data_3:  # true if not an empty list
        #Amp_Data_4[1023] = Amp_Data_4[0]
        Moffat_Model = models.Moffat1D(amplitude=Amp_Data_5[i],
                                        x_0 = Mean_Data_5[i],
                                        gamma = 2,
                                        alpha = Alpha_Data_5[i])
        Moffat_Model.x_0.fixed = True
        Moffat_Model.amplitude.fixed = True
        Moffat_Model.alpha.fixed = True
    Fit_Data_3.append(Fitting_Model(Moffat_Model, x, Image_Data_All2[:,i]))
    Fit_Data_3[0] = Fitting_Model(models.Moffat1D(amplitude=Amp_Data_5[0],
                                        x_0 = Mean_Data_5[0],
                                        gamma = 2, alpha = Alpha_Data_5[0]),
                                        x, Image_Data_All2[:,0])
x7 = np.empty((1024,4))
for n in range(1024):
    x7[n,:] = Fit_Data_3[n].parameters

Amp_Data_6 = x7[:,0]
Mean_Data_6 = x7[:,1]
Gamma_Data_6 = x7[:,2]
Alpha_Data_6 = x7[:,3]


#plt.plot(x, Image_Data_All2[:,0])
#plt.plot(x, Fit_Data_3[0](x))
#plt.plot(x, models.Moffat1D(amplitude = 1025, x_0 = -12.225, gamma = 1.3, alpha = 1.113)(x))
#plt.axis([0, 1, 0, 300])
#plt.show()

#plt.plot(Image_Data_All2[:,0]-models.Moffat1D(amplitude = 744.534, x_0 = 2.49348, gamma = 1.4, alpha = 0.989061)(x))
#plt.show()

#resi = Image_Data_All2[:,0]-models.Moffat1D(amplitude = 744.534, x_0 = 2.49348, gamma = 1.385, alpha = 0.989061)(x)
#print(np.max(resi)-np.min(resi))


FitMoffat_1 = np.empty((100,1024))

for n in range(1024):
    FitMoffat_1[:,n] = Fit_Data_3[n](x)

Residual_Moffat = Image_Data_All2 - FitMoffat_1


#Plots the residual for the first column
plt.plot(Residual_Moffat[:,0], 'ro')
#plt.plot(Residual_Moffat[:,1], 'ro')
#plt.axis([40, 60, -40, 20])
plt.show()

#Plot a 2D colour map residual for entire Moffat Image
plt.pcolormesh(np.log10(Residual_Moffat[0:99,0:1023]), cmap='RdBu')
plt.colorbar()
plt.show()




for n in range(1024):
    Fit_Data_3[n].x_0 =  Fit_Data_3[n].x_0 - Mean_Data_Fit[n]

for n in range(1024):
    Fit_Data_3[n].x_0 =  Fit_Data_3[n].x_0 + np.ones((1024,1))[n]*0.5

x6 = np.empty((1024,4))
for n in range(1024):
    x6[n,:] = Fit_Data_3[n].parameters

Mean_Data_5 = x6[:,1]







HR_x = np.linspace(-50, 50, 1000)

Moffat_Image = np.empty((100,1024))
Moffat_Image_HR = np.empty((1000,1024))

for n in range(1024):
    Moffat_Image[:,n] = Fit_Data_3[n](x)

for n in range(1024):
    Moffat_Image_HR[:,n] = Fit_Data_3[n](HR_x)


plt.plot(Moffat_Image[:,0])
plt.show()

plt.plot(Moffat_Image_HR[:,0])
plt.show()

Moffat_Image_2 = Moffat_Image.sum(axis = 1)

plt.plot(Moffat_Image_2)
plt.show()

plt.pcolormesh(Moffat_Image_HR, cmap='RdBu')
plt.colorbar()
plt.show()





#Redo Moffat fit to fix the gamma parameter which is not fitted in a single loop

Fit_Data_6 = []

#for i in range(0, Image_Data_All2.shape[1]):
#    Fit_Data_3.append(Fitting_Model(Moffat_Model, x, Image_Data_All2[:,i]))

for i in range(0, Image_Data_All2.shape[1]):
    if Fit_Data_3:  # true if not an empty list
        #Amp_Data_4[1023] = Amp_Data_4[0]
        Moffat_Model = models.Moffat1D(amplitude=Amp_Data_5[i],
                                        x_0 = Mean_Data_5[i],
                                        gamma = 1.3,
                                        alpha = Alpha_Data_5[i])
        Moffat_Model.x_0.fixed = True
        Moffat_Model.amplitude.fixed = False
        Moffat_Model.alpha.fixed = False
    Fit_Data_6.append(Fitting_Model(Moffat_Model, HR_x, Moffat_Image_HR[:,i]))
    Fit_Data_6[0] = Fitting_Model(models.Moffat1D(amplitude=Amp_Data_5[0],
                                        x_0 = Mean_Data_5[0],
                                        gamma = 1.3, alpha = Alpha_Data_5[0]),
                                        HR_x, Moffat_Image_HR[:,0])

x8 = np.empty((1024,4))
for n in range(1024):
    x8[n,:] = Fit_Data_6[n].parameters

Amp_Data_7 = x8[:,0]
Mean_Data_7 = x8[:,1]
Gamma_Data_7 = x8[:,2]
Alpha_Data_7 = x8[:,3]



















#Manual Method for Polynomial Sigma Clipping of Mean Position
Polydegree = 2
Mean_Range = np.linspace(0, 1023, 1024)
Polytest = np.polyfit(Mean_Range, Mean_Data_2, Polydegree)
Polyfit_Function = Polytest[0]*Mean_Range**2 + Polytest[1]*Mean_Range**1 + Polytest[2]
Mean_Residual = (Mean_Data_2 - Polyfit_Function)
Mean_Residual = sigma_clip(Mean_Residual, sigma = 5)
Mean_Residual = sigma_clip(Mean_Residual, sigma = 3)
Mean_Data_Clipped = Mean_Residual + Polyfit_Function
Mean_Data_Clipped = Mean_Data_Clipped[~Mean_Data_Clipped.mask]
Mean_Range_2 = np.linspace(0, Mean_Data_Clipped.shape[0]-1, Mean_Data_Clipped.shape[0])
Polytest_2 = np.polyfit(Mean_Range_2, Mean_Data_Clipped, Polydegree)
Polyfit_Function_2 = Polytest_2[0]*Mean_Range_2**2 + Polytest_2[1]*Mean_Range_2**1 + Polytest_2[2]
Mean_Residual_2 = (Mean_Data_Clipped - Polyfit_Function_2)
Mean_Residual_2 = sigma_clip(Mean_Residual_2, sigma = 5)
Mean_Residual_2 = sigma_clip(Mean_Residual_2, sigma = 3)
Mean_Data_Clipped_2 = Mean_Residual_2 + Polyfit_Function_2
Mean_Data_Clipped_2 = Mean_Data_Clipped_2[~Mean_Data_Clipped_2.mask]
Mean_Range_3 = np.linspace(0, Mean_Data_Clipped_2.shape[0]-1, Mean_Data_Clipped_2.shape[0])
Polytest_3 = np.polyfit(Mean_Range_3, Mean_Data_Clipped_2, Polydegree)
Polyfit_Function_3 = Polytest_3[0]*Mean_Range_3**2 + Polytest_3[1]*Mean_Range_3**1 + Polytest_3[2]
Mean_Residual_3 = (Mean_Data_Clipped_2 - Polyfit_Function_3)
Mean_Residual_3 = sigma_clip(Mean_Residual_3, sigma = 5)
Mean_Residual_3 = sigma_clip(Mean_Residual_3, sigma = 3)
Mean_Data_Clipped_3 = Mean_Residual_3 + Polyfit_Function_3
Mean_Data_Clipped_3 = Mean_Data_Clipped_3[~Mean_Data_Clipped_3.mask]
Mean_Range_4 = np.linspace(0, Mean_Data_Clipped_3.shape[0]-1, Mean_Data_Clipped_3.shape[0])
Polytest_4 = np.polyfit(Mean_Range_4, Mean_Data_Clipped_3, Polydegree)
Polyfit_Function_4 = Polytest_4[0]*Mean_Range_4**2 + Polytest_4[1]*Mean_Range_4**1 + Polytest_4[2]
Mean_Residual_4 = (Mean_Data_Clipped_3 - Polyfit_Function_4)
Mean_Residual_4 = sigma_clip(Mean_Residual_4, sigma = 5)
Mean_Residual_4 = sigma_clip(Mean_Residual_4, sigma = 3)
Mean_Data_Clipped_4 = Mean_Residual_4 + Polyfit_Function_4
Mean_Data_Clipped_4 = Mean_Data_Clipped_4[~Mean_Data_Clipped_4.mask]
Mean_Range_5 = np.linspace(0, Mean_Data_Clipped_4.shape[0]-1, Mean_Data_Clipped_4.shape[0])
Polytest_5 = np.polyfit(Mean_Range_5, Mean_Data_Clipped_4, Polydegree)
Polyfit_Function_5 = Polytest_5[0]*Mean_Range_5**2 + Polytest_5[1]*Mean_Range_5**1 + Polytest_5[2]
Mean_Residual_5 = (Mean_Data_Clipped_4 - Polyfit_Function_5)
Mean_Residual_5 = sigma_clip(Mean_Residual_5, sigma = 5)
Mean_Residual_5 = sigma_clip(Mean_Residual_5, sigma = 3)
Mean_Data_Clipped_5 = Mean_Residual_5 + Polyfit_Function_5
Mean_Data_Clipped_5 = Mean_Data_Clipped_5[~Mean_Data_Clipped_5.mask]
Mean_Range_6 = np.linspace(0, Mean_Data_Clipped_5.shape[0]-1, Mean_Data_Clipped_5.shape[0])
Polytest_6 = np.polyfit(Mean_Range_6, Mean_Data_Clipped_5, Polydegree)
Polyfit_Function_6 = Polytest_6[0]*Mean_Range_6**2 + Polytest_6[1]*Mean_Range_6**1 + Polytest_6[2]
Mean_Residual_6 = (Mean_Data_Clipped_5 - Polyfit_Function_6)
Mean_Residual_6 = sigma_clip(Mean_Residual_6, sigma = 5)
Mean_Residual_6 = sigma_clip(Mean_Residual_6, sigma = 3)
Mean_Data_Clipped_6 = Mean_Residual_6 + Polyfit_Function_6
Mean_Data_Clipped_6 = Mean_Data_Clipped_6[~Mean_Data_Clipped_6.mask]
Mean_Range_7 = np.linspace(0, Mean_Data_Clipped_6.shape[0]-1, Mean_Data_Clipped_6.shape[0])
Polytest_7 = np.polyfit(Mean_Range_7, Mean_Data_Clipped_6, Polydegree)
Polyfit_Function_7 = Polytest_7[0]*Mean_Range_7**2 + Polytest_7[1]*Mean_Range_7**1 + Polytest_7[2]
Mean_Residual_7 = (Mean_Data_Clipped_6 - Polyfit_Function_7)
Mean_Residual_7 = sigma_clip(Mean_Residual_7, sigma = 5)
Mean_Residual_7 = sigma_clip(Mean_Residual_7, sigma = 3)
Mean_Data_Clipped_7 = Mean_Residual_7 + Polyfit_Function_7
Mean_Data_Clipped_7 = Mean_Data_Clipped_7[~Mean_Data_Clipped_7.mask]


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





