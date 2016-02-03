#Standard Imports
import scipy as sp
import scipy.stats as ss
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.neighbors import KernelDensity as KD

#Mongo Imports
import mongoengine     # django-like queries to mongo
import xmongo          # this defines the collection schemas
import pprint          # for pretty printing examples
from util.munge.loading import Loader as L

# ###Function Definitions


'''Finds single points for each peak
Input: np.array formatted as a array of arrays 
        eg. [[value1, density1], [value2, density2]...]
output: List of peak points'''
def find_peaks(mountains, value_width):
    #Check for empty
    if (mountains == []):
        return []
    
    #Breaks up mountains matrix into a values vector, and densities vector
    m_array = np.array(mountains)
    value = m_array[:,0]
    density = m_array[:,1]
    
    #Seperate mountains by finding non-adjacent densities (w.r.t. to corresponding values)
    sep_mounts = [[mountains[0]]]    
    mountain_number = 0
    for i in range(1, value.size): 
        #New Mountain
        if ((value[i] - value[i-1]) > value_width*2.0):
            mountain_number += 1
            sep_mounts.append([mountains[i]])
        else: #Same Mountain 
            sep_mounts[mountain_number].append(mountains[i]) 
    
    #Peak Detector - Finds max points in value-connected mountain pieces
    result = []
    for m in sep_mounts:
        if len(m) == 1:
            result.append(m[0])
        else:
            #Set-up variables
            increasing = True
            m_arr = np.array(m)
            value = m_arr[:,0]
            density = m_arr[:,1]
            prev = density[0]
            
            #Find peaks
            for index, d in enumerate(density[1:]): 
                if (increasing == True) and (prev >= d):
                    result.append([value[index+1],prev])
                    increasing = False
                elif (increasing == False) and (prev < d):
                    increasing = True
                prev = d    
    
    return result

# assume times contains a vector of every time of every click on a single video

#Run peak detection
#p = detect_peaks(times, display_graph = True, KDE_bandwidth = 5)

#KDE Peak Detection Function (and Graphing)

'''Produces a list of peaks for a single video

Input parameters:
values - a Panda Series containing a list of the values over which you want to find peaks, no default
display_graph - default True, set to false if you do not wish to visualize the peaks
KDE_bandwidth - Sets how tightly the smooth curve fits to the data, default - 10
stds - sets how many standard devations from the mean is considered a peak, default 1.5
X_plot_intervals - sets how finely you wish to make the x-axis for the KDE smooth curve, default 1000.0
num_bins - sets the number of bins in the histogram if display_graph = True, default 80

output: This function returns a list of peak values from the input value series'''
def detect_peaks(values, display_graph = True, KDE_bandwidth = 2, stds = 1.5, X_plot_intervals = 1000.0, num_bins = 1000):
    
    #Find KDE Density Values: density values stored in dens, and "bins" values in X_plot
    X = values.values.reshape(values.values.size, 1)
    kde = KD(kernel='gaussian', bandwidth=KDE_bandwidth).fit(X)
    X_plot = np.linspace(X.min(), X.max(), X_plot_intervals)[:, np.newaxis]
    dens = np.exp(kde.score_samples(X_plot)) #produces KDE Density Values
    start = .005, stop = .3, step = .001
    nu_values = np.arange(.005,.3,.001)[:, np.newaxis]
    
    #Creates a set of [time,count] pairs for each peak, where a peak is 'stds' standard deviations above the mean
    mountains = [[X_plot[index], dens[index]] for index,d in enumerate(dens) if d > (stds*dens.std() + dens.mean())]
    mountain_tops = find_peaks(mountains, (X.max() - X.min()) / X_plot_intervals)
    
    #Display Histogram, KDE, and peaks
    if display_graph:
        fig = plt.figure(figsize=(14,6))
        ax1 = fig.add_subplot(111)
        #ax1.plot(X_plot[:, 0], dens, '-', color="Crimson", linewidth=2, label="Gaussian Kernel, bw=%.1f" % KDE_bandwidth )
        ax1.plot(nu_values[:, 0], dens, '-', color="Crimson", linewidth=2, label="Gaussian Kernel, bw=%.1f" % KDE_bandwidth )
        ax1.hist(X,normed=True,bins=num_bins,color="Silver",alpha=0.6)
        ax1.plot(*zip(*mountain_tops), marker='o', color='Blue', linewidth=10, ls='')
        ax1.hlines(1.5*dens.std() + dens.mean(), X_plot.min(), X_plot.max(), color='LightBlue', linestyle='--', linewidth = 2, label="Threshold, mean + %.1f*std" % stds)
        ax1.set_xlabel('Nu')
        ax1.set_ylabel('Density of Difference Rating')
        ax1.set_title('Kernel Density Estimation Peak Detection of Nu')
        ax1.legend()
        
    return [mountain_tops[index][0][0] for index in range(0,len(mountain_tops))]
