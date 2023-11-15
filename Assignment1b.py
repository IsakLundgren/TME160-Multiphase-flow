#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 15:28:09 2019

__authors__ = "Ananda S. Kannan and Niklas Hidman"
__institution__ = "Department of Mechanics and Maritime Sciences, Chalmers University of Technology, Sweden"
__copyright__ = "Copyright 2019, TME160: Multiphase flow"
__license__ = "GPL"
__version__ = "1.0"
__maintainers__ = "Ananda S. Kannan and Niklas Hidman"
__email__ = " ananda@chalmers.se and niklas.hidman@chalmers.se"

####################################################################################################################################
                                        "Py template to simulate a bubble column (in 2D)"

                Assumptions - 
 
                1)	Only 2D motion is studied (motion along the z-axis is assumed to be negligible)
                2)	Bubble starts from rest Vp(t = 0) = 0
                3)	The Continuous phase (water) is assumed to have a (parabolic) laminar velocity profile 
                4)	Only one-way coupling i.e. the background fluid affects the bubble motion and not vice-versa
                5)	5)	Inter-bubble (coalescence and break-up) and bubble-wall interactions (such as collisions etc.) are ignored
####################################################################################################################################
"""
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm
from scipy.integrate import quad
import numpy as np
import os
import sys

np.random.seed(12345)  # same seed for all random numbers

# %% Input for file storage##
simVer = "bubbleColumn"
# close all figures
plt.close("all")
# To create a directory and store the relevant post-processed files
# the file path is --> 'your run directory'/TME160/bubbleColumn
ppPath = os.path.dirname(os.path.dirname(__file__)) + "/" + simVer
imgPath = os.path.dirname(os.path.dirname(__file__)) + "/" + "img"
# Make directory  if it already doesn't exist
if not os.path.exists(ppPath):
    os.makedirs(ppPath)
# %%

"""
####################################################################################################################################
                                            Custom function definitions
####################################################################################################################################

"""


def getBubbleDia(meanDia, stdDevDia):
    """Samples a bubble diameter from a normal distribution"""
    """function I/P's : meanDia (bubble mean diameter)"""
    """function I/P's : stdDevDia (standard deviation of bubble mean diameter)"""
    """function O/P : bubble diameter (which is normally distributed)"""

    return np.random.normal(meanDia, stdDevDia)  # sample from a normal distribution


def injectBubble(meanDia, stdDevDia, bubbleList, bubbleMaxID, bubbleDiaArray,
                 bubbleXpos, bubbleInjectionTimeIndex, injectionXpos, timeIndex):
    """Injects bubbles and stores them in arrays"""
    """Arrays for storing bubble data"""
    """function I/P's : bubbleList (list of all bubbles injected)"""
    """function I/P's : bubbleMaxID (maximum value of the bubble ID among injected bubbles)"""
    """function I/P's : bubbleDiaArray (an array indexed based on bubbleID which contains bubble diameters of all injected bubbles)"""
    """function I/P's : bubbleXpos (an array indexed based on bubbleID which contains the x-positions of all injected bubbles)"""
    """NOTE: All bubbles injected at y = 0"""
    """function I/P's : bubbleInjectionTimeIndex (a list with all injection time indices)"""
    """injection specific data"""
    """function I/P's : injectionXpos (injector locations along x-axis )"""
    """function I/P's : timeIndex (time index of the injected bubbles)"""
    """function O/P's : newBubbleID, updatedBubbleList, bubbleDiaArray, bubbleXpos, bubbleInjectionTimeIndex"""

    newBubbleID = bubbleMaxID + 1  # update the bubble ID
    updatedBubbleList = bubbleList.append(newBubbleID)  # append new ID to exisitng list of bubble ID's
    try:
        bubbleDiaArray[newBubbleID] = getBubbleDia(meanDia,
                                                   stdDevDia)  # append new bubble diameter (sampled from normal distr.) to the existing array of bubble diameters
    except IndexError as e:

        sys.exit(
            '!!!You are trying to access an array position that does not exist. Probably the array is too small because n_tot_bubbles is too small. '
            '\n n_tot_bubbles is an integer that is used to allocate arrays and must be large enough to fit all bubbles injected during the simulation.\n'
            f'Error: {e}')

    bubbleXpos[
        timeIndex, newBubbleID] = injectionXpos  # append new bubble XPos to the existing array of bubble x-positions
    bubbleInjectionTimeIndex[newBubbleID] = timeIndex  # append new time index to exisitng list of time indices

    return newBubbleID, updatedBubbleList, bubbleDiaArray, bubbleXpos, bubbleInjectionTimeIndex


def fluidVelandGrad(py, x, b, mul):
    """Calculates the fluid velocity based on a provided laminar (parabolic) profile and calulate the velocity gradient along x-direction"""
    """function I/P's : py (pressure gradient driving the flow in Pa)"""
    """function I/P's : x (x-coordinate in the domain in m)"""
    """function I/P's : b (domain half width along x-dir in m)"""
    """function I/P's : mul (fluid dynamic viscosity in Pa.s)"""
    """function O/P's : fluidVelo = py*x/(2*mul)*(2*b-x); fluidXGrad = py/mul*(b-x)"""

    return py * x / (2 * mul) * (2 * b - x), py / mul * (
                b - x)  # calculate fluid velocity ; calculate the velocity gradient along the y-direction


def cl_tomiyama(Re, Eo, sig, rhoL, g):
    """Calculates the lift co-eff CL based on the Tomiyama model"""
    """function I/P's : Re (Bubble Reynold's number)"""
    """function I/P's : Re (Bubble Eotvo's number)"""
    """function I/P's : sig (Surface tension of the bubble)"""
    """function I/P's : rhoL (Fluid density in Kg/m3)"""
    """function I/P's : g (acceleration due to gravity in m2/s)"""
    """function O/P's : Cl_tom, Eoh (CL i.e. the lift co-efficient; Eo based on bubble horizontal diameter)"""

    Dbubble_eq = np.sqrt(Eo * sig / (rhoL * g))  # bubble equivalent diameter
    Dhorisontal = Dbubble_eq * (1 + 0.163 * Eo ** 0.757) ** (1 / 3)  # Horizontal bubble diameter Wellek et al. (1966)
    Eoh = rhoL * g * Dhorisontal ** 2 / sig
    feo = 0.00105 * Eoh ** 3 - 0.0159 * Eoh ** 2 - 0.0204 * Eoh + 0.474
    if Eoh < 4.0:
        Cl_tom = np.min([0.288 * np.tanh(0.121 * Re), feo])
    elif Eoh < 10.7:
        Cl_tom = feo
    else:  # assume cl at Eo=10.7 if outside validity range
        Eoh = 10.7
        Cl_tom = 0.00105 * Eoh ** 3 - 0.0159 * Eoh ** 2 - 0.0204 * Eoh + 0.474

    return Cl_tom


def areaAverageVoidFractionWithinBounds(y_up, y_down, n_bins, timeIndex, b):
    """Calculates a time averaged void fraction within given bounds in the y direction per binned section along the x-direction"""
    """function I/P's : y_up (upper y-bound)"""
    """function I/P's : y_down (lower y-bound)"""
    """function I/P's : n_bins (number of bins to section the x-axis of the domain)"""
    """function I/P's : timeIndex (time index of the injected bubbles)"""
    """function I/P's : b (domain half width along x-dir)"""
    """function O/P : voidFractionPerBin (time averaged void fraction per bin)"""

    voidFractionPerBin = np.zeros(n_bins)  # create a 1D array to store void fraction based on no. of bins
    for iBin in range(n_bins):  # compute areaaverage void frac for each bin, left to right (along x-direction)
        xMax = (iBin + 1) * 2.0 * b / n_bins  # max x-coord bound for current bin
        xMin = iBin * 2.0 * b / n_bins  # min x-coord bound for current bin
        bubblesInBound = list()  # list to store bubble IDs which correspond to the given bounded area
        for bubble in range(bubbleMaxID):  # check which bubbles that are inside the bounds at current timestep
            if (bubbleYpos[timeIndex, bubble] > y_down) and (bubbleYpos[timeIndex, bubble] < y_up):
                if (bubbleXpos[timeIndex, bubble] > xMin) and (bubbleXpos[timeIndex, bubble] < xMax):
                    bubblesInBound.append(bubble)

        # compute total area occupied by the bubbles in 2D
        voidArea = 0
        for bubble in bubblesInBound:
            voidArea = voidArea + bubbleDia[bubble] ** 2 * np.pi / 4.0

        totArea = (xMax - xMin) * (y_up - y_down)  # total area of the bin
        voidFractionPerBin[iBin] = voidArea / totArea  # void fraction per bin

    return voidFractionPerBin


"""
####################################################################################################################################
            Bubble column definition: Physical properties and domain description
####################################################################################################################################

"""

# Domain size
b = 0.025  # domain half width along x-dir (m)
L = 3.0  # domain length along y-dir (m)

# Physical properties
# Continous phase: Water @20 deg
rhoL = 1000.0  # density of water (kg/m3)
mul = 0.001  # dynamic viscosity of water (Pa.s)
sig = 0.073  # surface tension

# Dispersed phase: Air @20 deg
meanDia = 0.0058  # mean bubble diameter (m)
stdDevDia = 0.0015  # standard deviation in the bubble dia
rhoB = 1.2  # bubble density (kg/m3)
massMeanBubble = rhoB * 4 / 3 * np.pi * (meanDia / 2.0) ** 3  # mass of a mean dia bubble (kg)

# other constants
g = 9.82  # gravity
py = 1  # pressure gradient

"""
####################################################################################################################################
                        Bubble column definition: time step and injector related settings
####################################################################################################################################

"""
# time settings
dt = 0.1  # time step for the simulations (s) -- CHOOSE AN APPROPRIATE TIMESTEP
tEnd = 30  # simulation end time (s) -- SPECIFIED IN THE PROBLEM

# DON'T CHANGE THIS
n_timeSteps = int(np.ceil(tEnd / dt))  # total number of time steps
times = np.linspace(0, tEnd,
                    n_timeSteps)  # list with stored values for time indices (i.e a counter for n_timesteps: for e.g. times = [1,2, 3, 4 .... n_timeSteps])

# Injector settings
massFlowRateTot = 2.4e-6  # total mass flow rate of air (kg/s) -- TRY AND CHANGE THIS
n_nozzles = 6  # number of injection nozzles

massFlowRateNozzle = massFlowRateTot / n_nozzles  # mass flow rate per nozzle (kg/s)
bubbleInjectionFrequency = massFlowRateNozzle / massMeanBubble  # injection frequency required (1/s)

# total number of bubbles that will be injected during the simulation.
# Must be an integer, use for example the function int() to convert float to integer.
# The value must be larger than or equal to the total number of bubbles that is injected during the simulation. It is used to pre-allocate data arrays.
bubblesTotalMass = massFlowRateTot * tEnd


def integrand(d):
    return norm(meanDia, stdDevDia).pdf(d) / (rhoB * 4 / 3 * np.pi * (d / 2) ** 3)


''' Visualizing the PDF
plt.figure()
gamer = np.linspace(meanDia - 3.5 * stdDevDia, meanDia + 3.5 * stdDevDia, 1000)
plt.plot(gamer, integrand(gamer))
plt.show()
'''

n_expected_bubbles = bubblesTotalMass * quad(integrand, meanDia / 100, 2 * meanDia)[0]
n_tot_bubbles = int(1.15 * n_expected_bubbles)

"""
####################################################################################################################################
                        Storing bubble data and other related parameters as arrays
####################################################################################################################################

"""
# Initialize all the relevant arrays for storing bubble data and fill them with zeros
bubbleXpos = np.zeros([n_timeSteps, n_tot_bubbles])  # time along with corresp. bubble x-positions
bubbleYpos = np.zeros([n_timeSteps, n_tot_bubbles])  # time along with corresp. bubble y-positions
bubbleVelXdir = np.zeros([n_timeSteps, n_tot_bubbles])  # time along with corresp. bubble x-velocities
bubbleVelYdir = np.zeros([n_timeSteps, n_tot_bubbles])  # time along with corresp. bubble y-velocities

# Initialize 1-D arrays with bubble injection related properties
bubbleDia = np.zeros([n_tot_bubbles])  # array with diameters of all injected bubbles
bubbleInjectionTimeIndex = np.zeros([n_tot_bubbles], dtype=int)  # array with lists of all time indices (for injection)
bubbleDeletionTimeIndex = np.zeros([n_tot_bubbles], dtype=int)  # array with lists of all time indices (for deletion)
invalidAssumptionBubbles = np.full([n_tot_bubbles],
                                   False)  # Boolean value to check if the conditions for spherical shape were ever broken
aliveBubblesID = list()  # list to store bubbles that are active in the simulation domain (bubbles are removed when they near a boundary:walls/outlet)

"""
####################################################################################################################################
                        Algorithm for the simulation (Forward Euler time integration of the particle equation)
####################################################################################################################################

"""
# Initializations
timeSinceInjection = 1
bubbleMaxID = -1
ti = 0  # initialize time

for t in times:

    # inject bubbles from all 3 nozzles if:
    if timeSinceInjection > 1.0 / bubbleInjectionFrequency:
        timeSinceInjection = 0
        for noz in range(n_nozzles):
            injectionPos = 2.0 * b / (n_nozzles + 1) * (noz + 1)
            newBubbleID, updatedBubbleList, bubbleDiaArray, bubbleXpos, bubbleInjectionTimeIndex = (
                injectBubble(meanDia, stdDevDia, aliveBubblesID, bubbleMaxID, bubbleDia,
                             bubbleXpos, bubbleInjectionTimeIndex, injectionPos, ti))
            bubbleMaxID = newBubbleID  # setting the maximum bubble ID (@ the relevant time step)
    else:
        timeSinceInjection = timeSinceInjection + dt

    # collect bubbles that move outside the domain height in this list
    bubbleRemoveList = list()
    # update pos for all live bubbles
    for bubbleID in aliveBubblesID:  # loop over all alive bubbles at current time step ti

        if ti == n_timeSteps - 1:  # no need to compute next vel and pos (TERMINATE SIMULATION)
            break

        D = bubbleDia[bubbleID]  # bubble diamater
        uBubble = bubbleVelXdir[ti, bubbleID]  # bubble X-velocity at current timestep
        vBubble = bubbleVelYdir[ti, bubbleID]  # bubble Y-velocity at current timestep

        massBubble = rhoB * 4.0 / 3 * np.pi * (D / 2.0) ** 3

        """Y-direction (vertical axis)"""

        # Get y-Velocity and gradient along x-direction
        Vy, dVdx = fluidVelandGrad(py, bubbleXpos[ti, bubbleID], b, mul)

        # Calculate relative velocity between bubble and the surrounding fluid
        Vrel = vBubble - Vy  # relative velocity along y-direction
        Re = rhoL * Vrel * D / mul  # Reynolds number
        Eo = g * np.abs(rhoL - rhoB) * D ** 2 / sig  # Eötvös number
        Cd = 24 / Re * (1 + 0.15 * Re ** 0.687)  # drag coefficient

        # Mark the bubble as bad if it does not satisfy spherical conditions
        if not invalidAssumptionBubbles[bubbleID]:
            if Re > 1:
                invalidAssumptionBubbles[bubbleID] = True
            elif Eo > 1:
                invalidAssumptionBubbles[bubbleID] = True

        # Calculate the forces on the bubble along the y-direction
        F_D = 1 / 2 * rhoL * D ** 2 * np.pi / 4 * Cd * np.abs(Vrel) * Vrel  # N
        F_g = -massBubble * g  # N
        F_P = massBubble * rhoL / rhoB * g  # N

        # Calculate the added mass
        m_added = 1 / 2 * massBubble * rhoL / rhoB

        # Store time index when bubble is injected: used for computing the history force
        injectionTimeIndex = np.where(bubbleXpos[:, bubbleID] > 0.0)[0][0]
        # the history force Fhist is computed for you as:
        injTime = times[injectionTimeIndex]
        if (t - injTime) > 0.0:
            mhist = np.sqrt(rhoL * mul * np.pi) * massBubble / (rhoB * D)
            Fhist = mhist * Vrel / np.sqrt(0.5 * (t - injTime))
        else:
            Fhist = 0

        FtotY = F_D + F_g + F_P + Fhist  # total force on the bubble along y-direction
        totMass = massBubble + m_added  # total mass of the bubble + mass of the fluid carried by the bubble

        # Calculate bubble y-velocity at the new time-index ti+1: Forward Euler
        bubbleVelYdir[ti + 1, bubbleID] = bubbleVelYdir[ti, bubbleID] + dt * FtotY / totMass

        # Calculate new bubble y-position at the new time-index ti+1:
        bubbleYpos[ti + 1, bubbleID] = bubbleYpos[ti, bubbleID] + bubbleVelYdir[ti + 1, bubbleID] * dt

        # Domain-treatment
        # if bubble pos is above domain height L, remove from alive bubble list
        if bubbleYpos[ti + 1, bubbleID] > L:
            bubbleRemoveList.append(bubbleID)
            bubbleDeletionTimeIndex[bubbleID] = ti

        """X-direction (horizontal axis)"""

        Cl = cl_tomiyama(Re, Eo, sig, rhoL, g)  # Calculate lift co-efficient

        # Calculate the forces on the bubble along x-direction
        # TODO: must be filled

        FtotX = 1  # total force on the bubble along x-direction

        # Calculate bubble x-velocity at the new time-index ti+1: Forward Euler
        # TODO: must be filled
        bubbleVelXdir[ti + 1, bubbleID] = 1

        # Calculate new bubble x-position at the new time-index ti+1:
        # TODO: must be filled
        bubbleXpos[ti + 1, bubbleID] = bubbleXpos[ti, bubbleID] + bubbleVelXdir[ti + 1, bubbleID] * dt

        # Wall-treatment
        # if bubble pos is at one radius distance from a wall and x-vel towards it, set x-vel to 0 and bubble pos to previous pos.
        if (bubbleXpos[ti + 1, bubbleID] < D / 2.0) and (bubbleVelXdir[ti, bubbleID] < 0.0):
            bubbleVelXdir[ti, bubbleID] = 0.0
            bubbleXpos[ti + 1, bubbleID] = bubbleXpos[ti, bubbleID]
        elif (bubbleXpos[ti + 1, bubbleID] > (2 * b - D / 2.0)) and (bubbleVelXdir[ti, bubbleID] > 0.0):
            bubbleVelXdir[ti, bubbleID] = 0.0
            bubbleXpos[ti + 1, bubbleID] = bubbleXpos[ti, bubbleID]

    # remove bubble ID's from list aliveBubblesID (that have left the domain)
    for bubble in bubbleRemoveList:
        aliveBubblesID.remove(bubble)

    # update time index (go to the next timestep)
    ti = ti + 1

"""
####################################################################################################################################
                        Post-processing the bubble data: relevant plots
####################################################################################################################################

"""
# %%
# Plotting bubble trajectories (colored by bubble dia): FIGURE 1
VyToPlot, dVdxToPlot = fluidVelandGrad(py, np.linspace(0, 2 * b, 100), b, mul)

fig, ax1 = plt.subplots(figsize=(7, 6))
ax1.set_xlabel('x-coord')
ax1.set_ylabel('y-coord')
for bubble in range(bubbleMaxID):
    tStart = bubbleInjectionTimeIndex[bubble] + 1
    tEnd = bubbleDeletionTimeIndex[bubble] - 1
    ax1.plot(bubbleXpos[tStart:tEnd, bubble], bubbleYpos[tStart:tEnd, bubble],
             color=cm.jet(bubbleDia[bubble] / np.max(bubbleDia)))

ax2 = ax1.twinx()
ax2.set_ylabel('Velocity profile')
ax2.plot(np.linspace(0, 2 * b, 100), VyToPlot, '--')
ax1.set_title('Bubble trajectory (colored by bubble size)')
plt.grid(True)
figName = "BubbleTrajectories.png"
plt.savefig(os.path.join(imgPath, figName), dpi=250, bbox_inches='tight')
plt.show()

# Plotting time averaged void fractions
binsInXdir = 10  # number of bins along x-direction
dy = 0.02  # delta y (height of the averaging area along y-direction)

# Create 3 distinct regions along the bubble column for averaging (@y = 0.3m; y = 1.0m and y = 2.0m)
avVoidFracArea1 = np.zeros([n_timeSteps, int(binsInXdir)])  # @y = 0.3m
avVoidFracArea2 = np.zeros([n_timeSteps, int(binsInXdir)])  # @y = 1.0m
avVoidFracArea3 = np.zeros([n_timeSteps, int(binsInXdir)])  # @y = 2.0m

for i in range(n_timeSteps):
    avVoidFracArea1[i, :] = areaAverageVoidFractionWithinBounds(0.3, 0.3 - dy, binsInXdir, i, b)
    avVoidFracArea2[i, :] = areaAverageVoidFractionWithinBounds(1, 1 - dy, binsInXdir, i, b)
    avVoidFracArea3[i, :] = areaAverageVoidFractionWithinBounds(2, 2 - dy, binsInXdir, i, b)

# time averaging
startAverTimeIndex = int(bubbleDeletionTimeIndex[
                             0] * 1.15)  # skipping first time-instants (why is this done?)-- CHOOSE AN APPROPRIATE INTERVAL
timeAverageOverBins1 = np.mean(avVoidFracArea1[startAverTimeIndex:-1, :], axis=0)  # @y = 0.3m
timeAverageOverBins2 = np.mean(avVoidFracArea2[startAverTimeIndex:-1, :], axis=0)  # @y = 1.0m
timeAverageOverBins3 = np.mean(avVoidFracArea3[startAverTimeIndex:-1, :], axis=0)  # @y = 2.0m

# Plot of area averaged void fraction over time: FIGURE 2
plt.figure(figsize=(7, 6))
plt.rcParams.update({'font.size': 13})
plt.rcParams["font.family"] = "serif"
x_pos_bins = [i * 2 * b / binsInXdir + 2 * b / (2 * binsInXdir) for i in range(binsInXdir)]
plt.plot(x_pos_bins, timeAverageOverBins1, '-*')
plt.plot(x_pos_bins, timeAverageOverBins2, '-*')
plt.plot(x_pos_bins, timeAverageOverBins3, '-*')
plt.xlabel('x-pos')
plt.ylabel('time av void fraction')
plt.legend(['y = 0.3m', 'y = 1.0m', 'y = 2.0m'])
plt.grid(True)
figName = "AreaAveragedVoidFraction.png"
plt.savefig(os.path.join(imgPath, figName), dpi=250, bbox_inches='tight')
plt.show()

# bubble pos at a time instant: FIGURE 3
plt.figure(figsize=(7, 6))
plt.rcParams.update({'font.size': 13})
plt.rcParams["font.family"] = "serif"
timeIndexToPlot = int(bubbleDeletionTimeIndex[0] * 1.15)  # CHOOSE AN APPROPRIATE VALUE
for bubble in range(bubbleMaxID):
    if bubbleYpos[timeIndexToPlot, bubble] > 0.0:
        plt.plot(bubbleXpos[timeIndexToPlot, bubble], bubbleYpos[timeIndexToPlot, bubble], 'o',
                 color=cm.jet(bubbleDia[bubble] / np.max(bubbleDia)))
plt.ylim([0, L])
plt.xlabel('x-pos')
plt.ylabel('y-pos')
plt.title('Bubble pos at a given time instant (colored by bubble size)')
figName = "BubblePosAtAGivenTimeInstant.png"
plt.savefig(os.path.join(imgPath, figName), dpi=250, bbox_inches='tight')
plt.show()
