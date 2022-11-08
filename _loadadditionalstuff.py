#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 10:04:25 2022

@author: dlinhardt
"""

from glob import glob
from os import path

import numpy as np
from scipy.io import loadmat


#----------------------------------------------------------------------------#
def loadStim(self, buildTC=True):
    """
    loads the stimulus images from the reults.mat and builds the model TC
    for all positions on the stimulus

    Args:
        buildTC (bool, optional): Should we calculate the model TC. Defaults to True.
    """
    
    if self._dataFrom == 'mrVista':
        self.window = self.params['stim']['stimwindow'].flatten().reshape(101, 101).astype('bool')
        self.stimImages = self.params['analysis']['allstimimages'].T
        self.stimImagesUnConv = self.params['analysis']['allstimimages_unconvolved'].T

        if buildTC:
            self.X0 = self.params['analysis']['X']
            self.Y0 = self.params['analysis']['Y']

            pRF = np.exp(((self.x[:, None] - self.X0[None, :])**2 + (self.y[:, None] - self.Y0[None, :])**2)
                         / (-2 * self.s[:, None]**2))

            self.TC = pRF.dot(self.stimImages)

    else:
        size = np.sqrt(len(self.params[0]['stim']['stimwindow'])).astype(int)
        self.window = self.params[0]['stim']['stimwindow'].flatten().reshape(size, size).astype('bool')
        self.stimImages = self.params[0]['analysis']['allstimimages'].T
        self.stimImagesUnConv = self.params[0]['analysis']['allstimimages_unconvolved'].T

        if buildTC:
            self.X0 = self.params[0]['analysis']['X']
            self.Y0 = self.params[0]['analysis']['Y']

            pRF = np.exp(((self.x[:, None] - self.X0[None, :])**2 + (self.y[:, None] - self.Y0[None, :])**2)
                         / (-2 * self.s[:, None]**2))

            self.TC = pRF.dot(self.stimImages)

#----------------------------------------------------------------------------#
def loadTC(self, doMask=True):
    """
    loads the tSeries mat file used by mrVista containing all the input
    TC. Only possible for data from mrVista.
    TODO: implement this for from_docker data, its found in the results
    folder as _testdata.nii.gz

    Args:
        doMask (bool, optional): This is probalblz not working I guess, should apply the ROI and VarExp masks. Defaults to True.

    Returns:
        self.voxelTC: the input TC
    """
    
    if self._dataFrom == 'mrVista':
        TCs = loadmat(glob(path.join(self._baseP, self._study, 'subjects', self.subject, self.session,
                                     'mrVista', self._analysis, 'Gray/*/TSeries/Scan1/tSeries1.mat'))[0],
                      simplify_cells=True)['tSeries']

        if not doMask:
            self.voxelTC = TCs.T
        else:
            if self.isROIMasked:
                TCs = TCs[:, self.roiMsk]
            if self._isVarExpMasked:
                TCs = TCs[:, self._varExpMsk]
            # TCs = (TCs / TCs.mean(0) - 1) * 100
            self.voxelTC = TCs.T

        return self.voxelTC
    else:
        raise Warning('[loadStim] is only possible with data from mrVista!')


#----------------------------------------------------------------------------#
def loadJitter(self):
    """
    loads the EyeTracker file when converted to .mat.
    Needs to be passed to the analysis when data from mrVista (jitter)
    Needs to be in BIDS/sourcedata/etdata/ in BIDS format for docker

    Returns:
        self.jitterX, self.jitterY: both dimensions jitter
    """
    
    if self._dataFrom == 'mrVista':
        self.jitterP = path.join(self._baseP, self._study, 'subjects', self.subject,
                                 self.session, 'mrVista', self._analysis, 'Stimuli',
                                 f'{self.subject}_{self.session}_{self._analysis.split("ET")[0][:-1]}_jitter.mat')

        if path.exists(self.jitterP):
            jitter = loadmat(self.jitterP, simplify_cells=True)

            self.jitterX = jitter['x'] - jitter['x'].mean()
            self.jitterY = jitter['y'] - jitter['y'].mean()

            return self.jitterX, self.jitterY
        else:
            raise Warning('No jitter file found!')

    elif self._dataFrom == 'docker':
        self.jitterP = path.join(self._baseP, self._study, 'BIDS', 'etdata', self.subject, self.session,
                                 f'{self.subject}_{self.session}_task-{self._task.split("-")[0]}_run-0{self._run}_gaze.mat')

        if path.exists(self.jitterP):
            jitter = loadmat(self.jitterP, simplify_cells=True)

            self.jitterX = jitter['x'] - jitter['x'].mean()
            self.jitterY = jitter['y'] - jitter['y'].mean()

            return self.jitterX, self.jitterY
        else:
            raise Warning('No jitter file found!')


#----------------------------------------------------------------------------#
def loadRealign(self):
    """
    loads realignment parameter and calculated framewise displacement
    For nor only working with custom_preproc data where rp_avols.txt is available

    """

    if self._dataFrom == 'mrVista':
        # framewise displacement
        displ  = np.loadtxt(path.join(self.niiFolder, 'rp_avols.txt'))
        displ  = np.abs(np.diff(displ * [1, 1, 1, 50, 50, 50], axis=0))
        self.dispS = displ.sum()
        self.dispM = displ.mean()
    else:
        raise Warning('[loadStim] is only possible with data from mrVista!')
