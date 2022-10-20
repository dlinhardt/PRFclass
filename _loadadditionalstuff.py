#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 10:04:25 2022

@author: dlinhardt
"""

import numpy as np
from os import path
from glob import glob
from scipy.io import loadmat


#----------------------------------------------------------------------------#
# load the stim images and build TCs

def loadStim(self, buildTC=True):
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
# load the TCs file


def loadTC(self, doMask=True):
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
# load the jitter file

def loadJitter(self):
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
# load the realignment parameters

def loadRealign(self):
    # framewise displacement
    displ  = np.loadtxt(path.join(self.niiFolder, 'rp_avols.txt'))
    displ  = np.abs(np.diff(displ * [1, 1, 1, 50, 50, 50], axis=0))
    self.dispS = displ.sum()
    self.dispM = displ.mean()