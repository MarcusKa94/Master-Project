# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 17:23:30 2018

@author: Marcus Karlsson
"""

from astropy.io import fits
from astropy.stats import sigma_clip
import numpy as np
from matplotlib import pyplot as plt
from astropy.modeling import models, fitting
import os
import scipy

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

Image_Pixel_Data = np.nan_to_num(Image_Data_All2)

#Create the Lorentz and Moffat fits and create Fit_Data which contains the parameters for the models
x = np.linspace(-50,50,100)

Fitting_Model = fitting.LevMarLSQFitter()

Fit_Data_1 = []

##Normal Lorentz Model Fitting

Lorentz_Model = models.Lorentz1D(amplitude = 1000, x_0 = 0, fwhm = 1)
Lorentz_Model.x_0.fixed = False

for i in range(0, Image_Pixel_Data.shape[1]):
    if Fit_Data_1:  # true if not an empty list
        Lorentz_Model = models.Lorentz1D(amplitude = 700, x_0 = 0, fwhm = 1)
        Lorentz_Model.x_0.fixed = False
    Fit_Data_1.append(Fitting_Model(Lorentz_Model, x, Image_Pixel_Data[:,i]))


p1 = np.empty((1024,3))
for n in range(1024):
    p1[n,:] = Fit_Data_1[n].parameters

Ampl_Data_1 = p1[:,0]
Mean_Data_1 = p1[:,1]
FWHM_Data_1 = p1[:,2]

FitLorentz_1 = np.empty((100,1024))

for n in range(1024):
    FitLorentz_1[:,n] = Fit_Data_1[n](x)

Residual_Lorentz = Image_Pixel_Data - FitLorentz_1


#Polynomial fitting to Mean_Data_1 which creates the array Mean_Data_Fit
#Mean_Data_Fit is the ideal location of the mean of the fit

Mean_Data_Clipped = np.zeros((1023,))
Mean_Data_Residual = np.zeros((1024,))

while len(Mean_Data_Clipped) < len(Mean_Data_Residual):
    if len(Mean_Data_Clipped) < len(Mean_Data_Residual):
        Polydegree = 2
        Mean_Range = np.linspace(0, len(Mean_Data_1)-1, len(Mean_Data_1))
        Polytest = np.polyfit(Mean_Range, Mean_Data_1, Polydegree)
        Polyfit_Function = Polytest[0]*Mean_Range**2 + Polytest[1]*Mean_Range**1 + Polytest[2]
        Mean_Data_Residual = (Mean_Data_1 - Polyfit_Function)
        Mean_Data_Residual = sigma_clip(Mean_Data_Residual, sigma = 5)
        Mean_Data_Residual = sigma_clip(Mean_Data_Residual, sigma = 3)
        Mean_Data_Clipped = Mean_Data_Residual + Polyfit_Function
        #Mean_Data_Clipped = Mean_Data_Clipped[~Mean_Data_Clipped.mask]
        Mean_Data_1 = Mean_Data_Clipped
    else:
        print ('Sigma Clipping Complete')

Mean_Poly_Fit = np.ma.polyfit(Mean_Range, Mean_Data_Clipped, Polydegree)
Mean_Range_Fit = np.linspace(0, Image_Pixel_Data.shape[1]-1, Image_Pixel_Data.shape[1])
Mean_Data_Fit = Mean_Poly_Fit[0]*Mean_Range_Fit**2 + Mean_Poly_Fit[1]*Mean_Range_Fit**1 + Mean_Poly_Fit[2]

plt.plot(Mean_Data_1)
plt.plot(Mean_Data_Fit)
plt.show()


#Normal Moffat Model Fitting

Moffat_Model = models.Moffat1D(amplitude = 1000, x_0 = 0, gamma = 1, alpha = 2)

Fit_Data_2 = []

for i in range(0, Image_Pixel_Data.shape[1]):
    if Fit_Data_2:  # true if not an empty list
        Moffat_Model = models.Moffat1D(amplitude=Ampl_Data_1[-1],
                                        x_0 = Mean_Data_Fit[i],
                                        gamma = 1.3, alpha = 0.9)
        Moffat_Model.x_0.fixed = True
    Fit_Data_2.append(Fitting_Model(Moffat_Model, x, Image_Pixel_Data[:,i]))
    Fit_Data_2[0] = Fitting_Model(models.Moffat1D(amplitude=Ampl_Data_1[0],
                                        x_0 = Mean_Data_Fit[0],
                                        gamma = 1.3, alpha = 0.9),
                                        x, Image_Pixel_Data[:,0])

p2 = np.empty((1024,4))
for n in range(1024):
    p2[n,:] = Fit_Data_2[n].parameters

Ampl_Data_2 = p2[:,0]
Mean_Data_2 = p2[:,1]
Gamma_Data_2 = p2[:,2]
Alpha_Data_2 = p2[:,3]


#Moffat Fit with a running median

Fit_Data_3 = []

for i in range(0, Image_Pixel_Data.shape[1]):
    if Fit_Data_3:  # true if not an empty list
        Moffat_Model = models.Moffat1D(amplitude=Ampl_Data_2[i],
                                        x_0 = Mean_Data_2[i],
                                        gamma = 1.3,
                                        alpha = Alpha_Data_2[i])
        Moffat_Model.x_0.fixed = True
        Moffat_Model.amplitude.fixed = True
        Moffat_Model.alpha.fixed = True
    Fit_Data_3.append(Fitting_Model(Moffat_Model, x, Image_Pixel_Data[:,i]))
    Fit_Data_3[0] = Fitting_Model(models.Moffat1D(amplitude=Ampl_Data_2[0],
                                        x_0 = Mean_Data_2[0],
                                        gamma = 1.3, alpha = Alpha_Data_2[0]),
                                        x, Image_Pixel_Data[:,0])
        
p3 = np.empty((1024,4))
for n in range(1024):
    p3[n,:] = Fit_Data_3[n].parameters

Ampl_Data_3 = p3[:,0]
Mean_Data_3 = p3[:,1]
Gamma_Data_3 = p3[:,2]
Alpha_Data_3 = p3[:,3]

FitMoffat_1 = np.empty((100,1024))

for n in range(1024):
    FitMoffat_1[:,n] = Fit_Data_3[n](x)

Residual_Moffat_1 = Image_Pixel_Data - FitMoffat_1

#Plot a 2D colour map residual for entire Moffat Image
plt.pcolormesh(np.log10(Residual_Moffat_1[0:99,0:1023]), cmap='RdBu')
plt.colorbar()
plt.show()


#Create an array with all of the columns at the same position

Fit_Data_Zero = Fit_Data_3

for n in range(1024):
    Fit_Data_Zero[n].x_0 =  Fit_Data_Zero[n].x_0 - Mean_Data_Fit[n]

for n in range(1024):
    Fit_Data_Zero[n].x_0 =  Fit_Data_Zero[n].x_0 + np.ones((1024,1))[n]*0.5

e1 = np.empty((1024,4))
for n in range(1024):
    e1[n,:] = Fit_Data_Zero[n].parameters

Mean_Data_Fixed = e1[:,1]

HR_Vector = np.linspace(-50, 50, 1000)

Moffat_Image = np.empty((100,1024))
Moffat_Image_HR = np.empty((1000,1024))

for n in range(1024):
    Moffat_Image[:,n] = Fit_Data_Zero[n](x)

for n in range(1024):
    Moffat_Image_HR[:,n] = Fit_Data_Zero[n](HR_Vector)

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


#Redo Moffat fit in High Res to fix the gamma parameter which is not fitted correctly

Fit_Data_4 = []

for i in range(0, Image_Pixel_Data.shape[1]):
    if Fit_Data_4:  # true if not an empty list
        Moffat_Model = models.Moffat1D(amplitude=Ampl_Data_3[i],
                                        x_0 = Mean_Data_3[i],
                                        gamma = 1.3,
                                        alpha = Alpha_Data_3[i])
        Moffat_Model.x_0.fixed = True
        Moffat_Model.amplitude.fixed = True
        Moffat_Model.alpha.fixed = True
    Fit_Data_4.append(Fitting_Model(Moffat_Model, HR_Vector, Moffat_Image_HR[:,i]))
    Fit_Data_4[0] = Fit_Data_3[0]

p4 = np.empty((1024,4))
for n in range(1024):
    p4[n,:] = Fit_Data_4[n].parameters

Ampl_Data_4 = p4[:,0]
Mean_Data_4 = p4[:,1]
Gamma_Data_4 = p4[:,2]
Alpha_Data_4 = p4[:,3]


FitMoffat_2 = np.empty((1000,1024))

for n in range(1024):
    FitMoffat_2[:,n] = Fit_Data_4[n](HR_Vector)

plt.pcolormesh(FitMoffat_2, cmap='RdBu')
plt.colorbar()
plt.show()


#Alpha_Data_Clipped = sigma_clip(Alpha_Data_4, sigma = 5)
#Alpha_Data_Clipped = sigma_clip(Alpha_Data_Clipped, sigma = 3)
#plt.plot(Alpha_Data_4)
#plt.plot(Alpha_Data_Clipped)


#plt.plot(Alpha_Data_4)
#plt.plot(range(1024), 1*np.sin())

#def alpha_test(x, a, b, c):
#    return a * np.sin(b * x) + c

#params, params_covariance = scipy.optimize.curve_fit(alpha_test, range(1024), Alpha_Data_4, p0=[10**-1.7, 0.02, 1])

#plt.figure(figsize=(12, 8))
#plt.plot(range(1024), Alpha_Data_4, label='Data')
#plt.plot(range(1024), alpha_test(range(1024), params[0], params[1], params[2]),
#         label='Fitted function')
#plt.plot(np.linspace(0,1023,1024), 10**-1.7 * np.sin(0.02 * np.linspace(0,1023,1024)) + 0.95)
#plt.legend(loc='best')
#plt.show()



#Fit_Image = []
#for i in range(0, Image_Data_All2.shape[1]):
#    Fit_Image.append(Fit_Data_1[i](x))
#Fitted_Image = np.array(Fit_Image)
#plt.pcolormesh(np.transpose(Fitted_Image))
#plt.show()
#plt.pcolormesh(Image_Data_All2)
#plt.show
#Image_Residual = Image_Data_All2 - np.transpose(Fitted_Image)
#plt.pcolormesh(Image_Residual, cmap='RdBu')
#plt.colorbar()
#plt.show()
#ax_x = np.linspace(1, 100, 100) #range(Image_Residual.shape[0])
#ax_y = np.linspace(1, 1024, 1024) #range(Image_Residual.shape[1])
#aX, aY = np.meshgrid(ax_x, ax_y)
#fig = plt.figure(figsize = (9, 6))
#ax = fig.gca(projection = '3d')
#ax = fig.add_subplot(111, projection='3d')
#ax.set_zlim(-150.0, 150.0)
#Image_Residual[13, 210] = 0
#Residual_3D = ax.plot_surface(aX, aY, np.transpose(Image_Residual), cmap=cm.coolwarm)
#fig.colorbar(Residual_3D, shrink=0.6, aspect=5)





