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
#Image_Data_All_2 is the one used for the fit, it includes 100 rows of pixels centered
#on the brightest line in the image
Image_Data_0 = fits1.data[:,0]
Image_Data2_0 = Image_Data_0[300:400]
Image_Data_All_1 = fits1.data[:,:]
Image_Data_All_2 = Image_Data_All_1[300:400]

Image_Pixel_Data = np.nan_to_num(Image_Data_All_2)

Image_Pixel_Data_HR = np.repeat(Image_Pixel_Data, 10, axis=0)

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

Moffat_Image_1 = np.empty((100,1024))
Moffat_Image_HR = np.empty((1000,1024))

for n in range(1024):
    Moffat_Image_1[:,n] = Fit_Data_Zero[n](x)

for n in range(1024):
    Moffat_Image_HR[:,n] = Fit_Data_Zero[n](HR_Vector)

Moffat_Image_2 = Moffat_Image_1.sum(axis = 1)


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


#Create a High-Resolution Image with the Pixel Data
      
Image_Pixel_Data_HR = np.repeat(Image_Pixel_Data, 10, axis=0)

#Aligned the pixel data to the centre of the image, uses Numpy.interp

Image_Pixel_Data_Aligned_1 = []

for n in range(0, 1024):
    Image_Pixel_Data_Aligned_1.append(np.interp(HR_Vector, HR_Vector - Mean_Data_Fit[n], Image_Pixel_Data_HR[:,n]))

Image_Pixel_Data_Aligned_2 = np.transpose(np.asarray(Image_Pixel_Data_Aligned_1))


#Calculate the Area under the curve of each column
#in the image and then normalize the area to 1

Pixel_Column_Area_1 = []

for n in range(0, 1024):
    Pixel_Column_Area_1.append(np.trapz(y = Image_Pixel_Data_HR[:,n]))

Pixel_Column_Area_2 = np.asarray(Pixel_Column_Area_1)

Image_Data_Normalized_1 = []

for n in range(0, 1024):
    Image_Data_Normalized_1.append(Image_Pixel_Data_HR[:,n]/Pixel_Column_Area_2[n])

Image_Data_Normalized_2 = np.transpose(np.asarray(Image_Data_Normalized_1))

#Test that the new area is 1 under each column of data

Pixel_Area_Test_1 = []

for n in range(0, 1024):
    Pixel_Area_Test_1.append(np.trapz(y = Image_Data_Normalized_2[:,n]))

Pixel_Area_Test_2 = np.asarray(Pixel_Area_Test_1)


#Perform the normalization of the aligned data

Pixel_Column_Area_3 = []

for n in range(0, 1024):
    Pixel_Column_Area_3.append(np.trapz(y = Image_Pixel_Data_Aligned_2[:,n]))

Pixel_Column_Area_4 = np.asarray(Pixel_Column_Area_3)

Image_Data_Aligned_Normalized_1 = []

for n in range(0, 1024):
    Image_Data_Aligned_Normalized_1.append(Image_Pixel_Data_Aligned_2[:,n]/Pixel_Column_Area_4[n])

Image_Data_Aligned_Normalized_2 = np.transpose(np.asarray(Image_Data_Aligned_Normalized_1))


#Align the original image with 100 rows and 1024 columns

Image_Aligned_1 = []

for n in range(0, 1024):
    Image_Aligned_1.append(np.interp(x, x - Mean_Data_Fit[n] + 0.5, Image_Pixel_Data[:,n]))

Image_Aligned_2 = np.transpose(np.asarray(Image_Aligned_1))


#Scale down method

Image_Pixel_Data_Aligned_3 = Image_Pixel_Data_Aligned_2[::10,:]


#Test to see the residual of the high-resolution image aligned and then scaled down and
#the original image that has only been aligned

Image_Alignment_Residual = Image_Pixel_Data_Aligned_3 - Image_Aligned_2

Image_Column_Average_1 = []
for n in range(1024):
    Image_Column_Average_1.append(np.mean(Image_Pixel_Data[:,n]))

Image_Column_Average_2 = np.asarray(Image_Column_Average_1)

"""
First method to find the master image, only take the mean of all rows of the aligned and
normalized image

Master_Image_1 = []

for n in range(1000):
    Master_Image_1.append(np.mean(Image_Data_Aligned_Normalized_2[n,:]))

Master_Image_2 = np.asarray(Master_Image_1)
"""

#Second method to determine the master image, take the mean of the five rows before and 
#after each row and leaving the first and last five columns blank.

Master_Image_1 = np.empty([1000,1024])

for i in range(1000):
    for n in range(1024):
        if n < 5:
            Master_Image_1[i,n] = np.mean(Image_Data_Aligned_Normalized_2[i,n:n+6])
        elif n > 1018:
            Master_Image_1[i,n] = np.mean(Image_Data_Aligned_Normalized_2[i,n-5:n])
        else:
            Master_Image_1[i,n] = np.mean(Image_Data_Aligned_Normalized_2[i,n-5:n+6])

"""
Master_Image_1 = np.empty([1000,1024])

for i in range(1000):
    for n in range(1024):
        if n < 5:
            Master_Image_1[i,n] = np.median(Image_Data_Aligned_Normalized_2[i,n:n+2])
        elif n > 1018:
            Master_Image_1[i,n] = np.median(Image_Data_Aligned_Normalized_2[i,n-1:n])
        else:
            Master_Image_1[i,n] = np.median(Image_Data_Aligned_Normalized_2[i,n-1:n+2])
"""

Master_Image_1 = np.empty([1000,1024])

for i in range(1000):
    for n in range(1024):
        if n < 5:
            Master_Image_1[i,n] = np.median(Image_Data_Aligned_Normalized_2[i,n:n+4])
        elif n > 1018:
            Master_Image_1[i,n] = np.median(Image_Data_Aligned_Normalized_2[i,n-3:n])
        else:
            Master_Image_1[i,n] = np.median(Image_Data_Aligned_Normalized_2[i,n-3:n+4])


Master_Image_Aligned_1 = []

for n in range(0, 1024):
    Master_Image_Aligned_1.append(np.interp(HR_Vector, HR_Vector + Mean_Data_Fit[n], Master_Image_1[:,n]))

Master_Image_Aligned_2 = np.transpose(np.asarray(Master_Image_Aligned_1))


from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve

kernel = Gaussian2DKernel(stddev=1)

Image_Convolved = convolve(Image_Pixel_Data_HR, kernel)


Image_Pixel_Data = np.nan_to_num(Image_Data_All_2)

Image_Pixel_Data_HR = np.repeat(Image_Pixel_Data, 10, axis=0)

#Image_Pixel_Data_HR_2 = np.repeat(Image_Pixel_Data, 10, axis=0)

for i in range(1000):
    for n in range(1024):
        if Image_Pixel_Data_HR[i,n] == 0:
            Image_Pixel_Data_HR[i,n] = np.mean(Image_Pixel_Data_HR[i,n-5:n+6])

#This piece of code will reduce some of the errors, ask Alexis if use it or not

"""
for i in range(1000):
    for n in range(1024):
        if n < 5:
            if np.abs(Image_Pixel_Data_HR[i,n]) > (np.sum(np.abs(Image_Pixel_Data_HR[i,n:n+11]))-np.abs(Image_Pixel_Data_HR[i,n])):
                Image_Pixel_Data_HR[i,n] = np.median(Image_Pixel_Data_HR[i,n:n+6])
        elif n > 1018:
            if np.abs(Image_Pixel_Data_HR[i,n]) > (np.sum(np.abs(Image_Pixel_Data_HR[i,n-10:n]))-np.abs(Image_Pixel_Data_HR[i,n])):
                Image_Pixel_Data_HR[i,n] = np.median(Image_Pixel_Data_HR[i,n-5:n])
        else:
            if np.abs(Image_Pixel_Data_HR[i,n]) > (np.sum(np.abs(Image_Pixel_Data_HR[i,n-5:n+6]))-np.abs(Image_Pixel_Data_HR[i,n])):
                Image_Pixel_Data_HR[i,n] = np.median(Image_Pixel_Data_HR[i,n-5:n+6])
"""

#Find the Chi-squared:

"""Convolved VERSION

from scipy.stats import chisquare

Chi_Sq_1 = []

for n in range(1024):
    Chi_Sq_1.append(chisquare(Master_Image_Aligned_2[:,n], Image_Convolved[:,n]))

Chi_Sq_2 = np.asarray(Chi_Sq_1)

Chi_Sq_3 = Chi_Sq_2[:,0]

Master_Image_Final = []

for n in range(1024):
    Master_Image_Final.append(Master_Image_Aligned_2[:,n]*Chi_Sq_3[n])

Master_Image_Final_2 = np.transpose(np.asarray(Master_Image_Final))

Image_Reduced = Master_Image_Final_2 - Image_Convolved

"""


from scipy.stats import chisquare

Chi_Sq_1 = []

for n in range(1024):
    Chi_Sq_1.append(chisquare(Master_Image_Aligned_2[:,n], Image_Pixel_Data_HR[:,n]))

Chi_Sq_2 = np.asarray(Chi_Sq_1)

Chi_Sq_3 = Chi_Sq_2[:,0]

Master_Image_Final_1 = []

for n in range(1024):
    Master_Image_Final_1.append(Master_Image_Aligned_2[:,n]*Chi_Sq_3[n])

Master_Image_Final_2 = np.transpose(np.asarray(Master_Image_Final_1))

#Change Master_Image_Final_2 to fix the interpolation problem

for i in range(1000):
    for n in range(1024):
        if (Master_Image_Final_2[i,n] != Master_Image_Final_2[i-1,n]
        and Master_Image_Final_2[i,n] != Master_Image_Final_2[i+1,n]):
            if (np.abs(np.abs(Master_Image_Final_2[i,n]) - np.abs(Master_Image_Final_2[i-5,n])) >
            np.abs(np.abs(Master_Image_Final_2[i,n]) - np.abs(Master_Image_Final_2[i+5,n]))):
                Master_Image_Final_2[i,n] = Master_Image_Final_2[i+5,n]
            else:
                Master_Image_Final_2[i,n] = Master_Image_Final_2[i-5,n]

Master_Image_Final_3 = Master_Image_Final_2[::10,:]

#Second solution to the interpolation problem

Master_Image_Final_4 = Master_Image_Final_2
shape = (100,1024)

sh = shape[0],Master_Image_Final_4.shape[0]//shape[0],shape[1],Master_Image_Final_4.shape[1]//shape[1]
Master_Image_Final_5 = Master_Image_Final_4.reshape(sh).mean(-1).mean(1)


"""def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    b = a.reshape(sh).mean(-1).mean(1)
    return b

rebin(Master_Image_Final_2_test, (100,1024))
"""

#Try to reduce Image_Pixel_Data and Image_Pixel_Data_HR here at the worst columns

Image_Pixel_Data = np.nan_to_num(Image_Data_All_2)

Image_Pixel_Data_HR = np.repeat(Image_Pixel_Data, 10, axis=0)

Image_Pixel_Data_HR_2 = np.repeat(Image_Pixel_Data, 10, axis=0)

count = 0
for i in range(100):
    for n in range(1024):
        if np.abs(Image_Pixel_Data[i,n]) > 5:
            if n < 5:
                if np.abs(Image_Pixel_Data[i,n]) > (np.sum(np.abs(Image_Pixel_Data[i,n:n+6]))-np.abs(Image_Pixel_Data[i,n])):
                    Image_Pixel_Data[i,n] = np.median(Image_Pixel_Data[i,n:n+6])
                    count += 1
            elif n > 1018:
                if np.abs(Image_Pixel_Data[i,n]) > (np.sum(np.abs(Image_Pixel_Data[i,n-5:n]))-np.abs(Image_Pixel_Data[i,n])):
                    Image_Pixel_Data[i,n] = np.median(Image_Pixel_Data[i,n-5:n])
                    count += 1
            else:
                if np.abs(Image_Pixel_Data[i,n]) > (np.sum(np.abs(Image_Pixel_Data[i,n-5:n+6]))-np.abs(Image_Pixel_Data[i,n])):
                    Image_Pixel_Data[i,n] = np.median(Image_Pixel_Data[i,n-5:n+6])
                    count += 1

for i in range(1000):
    for n in range(1024):
        if Image_Pixel_Data_HR[i,n] == 0:
            Image_Pixel_Data_HR[i,n] = np.mean(Image_Pixel_Data_HR[i,n-5:n+6])

count = 0
for i in range(1000):
    for n in range(1024):
        if np.abs(Image_Pixel_Data_HR[i,n]) > 5:
            if n < 5:
                if np.abs(Image_Pixel_Data_HR[i,n]) > (np.sum(np.abs(Image_Pixel_Data_HR[i,n:n+6]))-np.abs(Image_Pixel_Data_HR[i,n])):
                    Image_Pixel_Data_HR[i,n] = np.median(Image_Pixel_Data_HR[i,n:n+6])
                    count += 1
            elif n > 1018:
                if np.abs(Image_Pixel_Data_HR[i,n]) > (np.sum(np.abs(Image_Pixel_Data_HR[i,n-5:n]))-np.abs(Image_Pixel_Data_HR[i,n])):
                    Image_Pixel_Data_HR[i,n] = np.median(Image_Pixel_Data_HR[i,n-5:n])
                    count += 1
            else:
                if np.abs(Image_Pixel_Data_HR[i,n]) > (np.sum(np.abs(Image_Pixel_Data_HR[i,n-5:n+6]))-np.abs(Image_Pixel_Data_HR[i,n])):
                    Image_Pixel_Data_HR[i,n] = np.median(Image_Pixel_Data_HR[i,n-5:n+6])
                    count += 1

plt.plot(Image_Pixel_Data)#-Image_Pixel_Data_HR_2)
#plt.axis([0, 1024, -20, 20])
plt.show()

plt.plot(Image_Pixel_Data_HR)#-Image_Pixel_Data_HR_2)
#plt.axis([0, 1024, -20, 20])
plt.show()


Image_Reduced_1 = Master_Image_Final_2 - Image_Pixel_Data_HR

Image_Reduced_2 = Master_Image_Final_3 - Image_Pixel_Data

Image_Reduced_3 = Master_Image_Final_5 - Image_Pixel_Data

plt.plot(Image_Reduced_1[:,0:800])
plt.axis([400, 600, -100, 50])
plt.show()

plt.plot(Image_Reduced_2[:,0:800])
#plt.axis([40, 60, -100, 50])
plt.show()

plt.plot(Image_Reduced_3[:,0:800])
#plt.axis([40, 60, -100, 50])
plt.show()


Image_Reduced_Aligned_0 = []

for n in range(0, 1024):
    Image_Reduced_Aligned_0.append(np.interp(HR_Vector, HR_Vector - Mean_Data_Fit[n], Image_Reduced_1[:,n]))

Image_Reduced_Aligned_1 = np.transpose(np.asarray(Image_Reduced_Aligned_0))


Image_Reduced_Aligned_0 = []

for n in range(0, 1024):
    Image_Reduced_Aligned_0.append(np.interp(x, x - Mean_Data_Fit[n], Image_Reduced_2[:,n]))

Image_Reduced_Aligned_2 = np.transpose(np.asarray(Image_Reduced_Aligned_0))


Image_Reduced_Aligned_0 = []

for n in range(0, 1024):
    Image_Reduced_Aligned_0.append(np.interp(x, x - Mean_Data_Fit[n], Image_Reduced_3[:,n]))

Image_Reduced_Aligned_3 = np.transpose(np.asarray(Image_Reduced_Aligned_0))


plt.plot(Image_Reduced_Aligned_1)
plt.show()

plt.plot(Image_Reduced_Aligned_2)
plt.show()

plt.plot(Image_Reduced_Aligned_3)
plt.show()

#Bad Pixels on column 722, 825 and 200




plt.pcolormesh(Master_Image_Final_2[450:550,:]-Image_Pixel_Data_HR[450:550,:])
plt.colorbar()
plt.show()

plt.plot(Master_Image_Final_2[:,1020])
plt.plot(Image_Pixel_Data_HR[:,1020])
plt.axis([400, 600, 200, 1000])
plt.show()


plt.plot(Image_Reduced_1[:,100:200]) #Do for only lower columns
plt.axis([400, 600, -200, 200])
plt.show()












###Another Lorentz fit to extract amplitude

Fitting_Model = fitting.LevMarLSQFitter()

Fit_Data_5 = []

##Normal Lorentz Model Fitting

Lorentz_Model_2 = models.Lorentz1D(amplitude = 10, x_0 = 0, fwhm = 1)
Lorentz_Model.x_0.fixed = False

for n in range(1024):
    if Fit_Data_5:  # true if not an empty list
        Lorentz_Model_2 = models.Lorentz1D(amplitude = 10, x_0 = 0, fwhm = 1)
        Lorentz_Model_2.x_0.fixed = False
    Fit_Data_5.append(Fitting_Model(Lorentz_Model_2, HR_Vector, Image_Reduced_1[:,n]))


p5 = np.empty((1024,3))
for n in range(1024):
    p5[n,:] = Fit_Data_5[n].parameters

Ampl_Data_5 = p5[:,0]
Mean_Data_5 = p5[:,1]
FWHM_Data_5 = p5[:,2]

FitLorentz_5 = np.empty((1000,1024))

for n in range(1024):
    FitLorentz_5[:,n] = Fit_Data_5[n](HR_Vector)

plt.plot(np.abs(Ampl_Data_5[300:700]))
plt.show()

plt.plot(Ampl_Data_1)
plt.show()



"""
#Find what is the best value to multiply all columns with to best fit the original image

Flux_Multiplier = Image_Pixel_Data_Aligned_2/Master_Image_1

Flux_Multiplier = np.ma.masked_array(Flux_Multiplier, np.isnan(Flux_Multiplier))

Flux_Stdv = np.std(Flux_Multiplier)*10

#If a value is too large or small, replace with a median of the values close to it

for i in range(1000):
    for n in range(1024):
        if Flux_Multiplier[i,n] > Flux_Stdv or Flux_Multiplier[i,n] < -Flux_Stdv:
            #print(Flux_Multiplier[i,n])
            Flux_Multiplier[i,n] = np.median(Flux_Multiplier[i,n-50:n+51])


Flux_Test = np.convolve(Flux_Multiplier[10,:], Flux_Multiplier[:,10], 'same')

Flux_Multiplier_2 = Image_Pixel_Data/Master_Image_2

for i in range(100):
    for n in range(1024):
        if Flux_Multiplier_2[i,n] > Flux_Stdv or Flux_Multiplier_2[i,n] < -Flux_Stdv:
            Flux_Multiplier_2[i,n] = np.median(Flux_Multiplier_2[i,n-5:n+6])

#Find the image that is used to reduce the original image. Will be of size 100x1024 and
#no longer aligned to the centre or normalized.

Image_Reducer = Flux_Multiplier*Master_Image_1

Image_Reducer = 30000*Master_Image_1

#for n in range(1024):
#    Image_Reducer.append(Flux_Multiplier[:,n]*Master_Image_1)

#Image_Reducer_2 = np.transpose(np.asarray(Image_Reducer))

#Calculate the reduced image by subtracting the final master image from the original
#image, hopefully no periodic spiky pattern.

Image_Reduced = Image_Reducer - Image_Pixel_Data_Aligned_2

HR_Vector_2 = []

Master_Image_Aligned = []

for n in range(0, 1024):
    Master_Image_Aligned.append(np.interp(HR_Vector, HR_Vector + Mean_Data_Fit[n], Image_Reducer_2[:,n]))

Master_Image_Aligned_2 = np.transpose(np.asarray(Master_Image_Aligned))
"""





