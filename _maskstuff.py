#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 10:00:59 2022

@author: dlinhardt
"""
import numpy as np
import json
from scipy.io import loadmat
from os import path


#-----------------------------------MASKING-----------------------------------#
# limit the used voxels to the ROI

def maskROI(self, doV123=False, forcePath=False, area='V1', atlas='benson'):
    if self._dataFrom == 'mrVista':
        if isinstance(area, list):
            Warning('You can not give area lists for mrVista data!')
        if doV123:
            self._roiV1 = loadmat(path.join(self._baseP, self._study, 'ROIs', self.subject + '_V1_Combined.mat'),
                                  simplify_cells=True)['ROI']['coords'].astype('uint16')
            self._roiV2 = loadmat(path.join(self._baseP, self._study, 'ROIs', self.subject + '_V2_Combined.mat'),
                                  simplify_cells=True)['ROI']['coords'].astype('uint16')
            self._roiV3 = loadmat(path.join(self._baseP, self._study, 'ROIs', self.subject + '_V3_Combined.mat'),
                                  simplify_cells=True)['ROI']['coords'].astype('uint16')

            dtypeV1 = {'names': ['f{}'.format(i) for i in range(3)], 'formats': 3 * [self._roiV1.dtype]}
            dtypeV2 = {'names': ['f{}'.format(i) for i in range(3)], 'formats': 3 * [self._roiV2.dtype]}
            dtypeV3 = {'names': ['f{}'.format(i) for i in range(3)], 'formats': 3 * [self._roiV3.dtype]}

            self._roiIndV1 = np.intersect1d(self._coords.T.view(dtypeV1).T, self._roiV1.T.view(dtypeV1).T,
                                            return_indices=True)[1]
            self._roiIndV2 = np.intersect1d(self._coords.T.view(dtypeV2).T, self._roiV2.T.view(dtypeV2).T,
                                            return_indices=True)[1]
            self._roiIndV3 = np.intersect1d(self._coords.T.view(dtypeV3).T, self._roiV3.T.view(dtypeV3).T,
                                            return_indices=True)[1]

            m1 = np.zeros(self.x0.shape)
            m2 = np.zeros(self.x0.shape)
            m3 = np.zeros(self.x0.shape)
            m1[self._roiIndV1] = 1
            m2[self._roiIndV2] = 1
            m3[self._roiIndV3] = 1

            self._roiMsk = np.any((m1, m2, m3), 0)

            self._isROIMasked = 3

        else:
            if forcePath:
                self._roi = (loadmat(forcePath, simplify_cells=True)
                             ['ROI']['coords']).astype('uint16')
            else:
                self._roi = (loadmat(path.join(self._baseP, self._study, 'ROIs', self.subject + '_V1_Combined.mat'),
                                     simplify_cells=True)['ROI']['coords']).astype('uint16')

            dtype = {'names': ['f{}'.format(i) for i in range(3)], 'formats': 3 * [self._roi.dtype]}
            self._roiInd = np.intersect1d(self._coords.T.view(dtype).T, self._roi.T.view(dtype).T,
                                          return_indices=True)[1]

            self._roiMsk = np.zeros(self.x0.shape)
            self._roiMsk[self._roiInd] = 1

            # if hasattr(self, 'tValues'): self._tValues = self._tValues[self._msk]

            self._isROIMasked = 1

    elif self._dataFrom == 'docker':
        optsF = path.join(self._baseP, self._study, 'derivatives', self._prfanaMe, self._prfanaAn, 'options.json')
        with open(optsF, 'r') as fl:
            opts = json.load(fl)

        self._area  = area if isinstance(area, list) else [area]
        self._atlas = atlas if isinstance(atlas, list) else [atlas]

        self._roiMsk = np.zeros(self.x0.shape)

        self._roiIndBold     = np.empty(0, dtype=int)
        self._roiIndFsnative = np.empty(0, dtype=int)
        self._roiWhichHemi   = np.empty(0)

        hs = ['L', 'R'] if self._hemis == '' else [self._hemis]
        for ar in self._area:
            for at in self._atlas:
                self._areaJsons = []
                for h in hs:

                    areaJson = path.join(self._baseP, self._study, 'derivatives', 'prfprepare',
                                         f'analysis-{opts["prfprepareAnalysis"]}', self._subject, self._session, 'func',
                                         f'{self._subject}_{self._session}_hemi-{h.upper()}_desc-{ar}-{at}_maskinfo.json')
                    with open(areaJson, 'r') as fl:
                        maskinfo = json.load(fl)

                    self._areaJsons.append(maskinfo)

                    self._roiIndBold     = np.hstack((self._roiIndBold, maskinfo['roiIndBold']))
                    self._roiIndFsnative = np.hstack((self._roiIndFsnative, maskinfo['roiIndFsnative']))
                    self._roiWhichHemi   = np.hstack((self._roiWhichHemi, np.tile(h, len(maskinfo['roiIndFsnative']))))

                self._roiMsk[self._areaJsons[0]['roiIndBold']] = 1
                if len(self._areaJsons) == 2:
                    self._roiMsk[np.array(self._areaJsons[1]['roiIndBold']) + self._areaJsons[0]['thisHemiSize']] = 1

        self._hemi0size  = self._areaJsons[0]['thisHemiSize']
        self._roiIndBold[self._roiWhichHemi == 'R'] += self._hemi0size
        self._isROIMasked = 1


#----------------------------------------------------------------------------#
# remove all voxels that are infs or nans in varexp and apply VarExp threshold

def maskVarExp(self, varExpThresh, highThresh=None, spmPath=None):
    if spmPath:
        if not hasattr(self, 'tValues'):
            self.loadSPMmat(spmPath)

        self._spmMask = abs(self._tValues) > varExpThresh

    else:
        if not highThresh:
            with np.errstate(invalid='ignore'):
                self._varExpMsk = np.isfinite(self.varexp0) & (self.varexp0 >= varExpThresh)
        else:
            with np.errstate(invalid='ignore'):
                self._varExpMsk = np.isfinite(self.varexp0) & (self.varexp0 >= varExpThresh) & (self.varexp0 <= highThresh)

        # if hasattr(self, 'tValues'): self._tValues = self._tValues[self._msk]

    self._isVarExpMasked = varExpThresh


#----------------------------------------------------------------------------#
# mask with given eccentricity value for coveragePlot

def maskEcc(self, rad, doThresh=True):

    self._eccMsk = self.r0 < rad

    self._isEccMasked = rad

#----------------------------------------------------------------------------#
# mask with beta threshold and high sigmas as calculated from macfunc18-c01 & c02 zeroth as below


def maskBetaThresh(self, betaMax=50, doThresh=True, doBorderThresh=False, doHighBetaThresh=True):

    self._betaMsk = np.ones(self.x0.shape)

    if doBorderThresh:
        self._betaMsk = np.all((self._betaMsk, self.s0 < self.r0 * 0.26868116 + 1.61949558), 0)
    if doHighBetaThresh:
        self._betaMsk = np.all((self._betaMsk, self.beta0 < betaMax), 0)

    self._isBetaMasked = 1 if doHighBetaThresh else 2 if doBorderThresh else 3

#----------------------------------------------------------------------------#
# calculate the convolved mask


def _calcMask(self, doROIMsk=True, doVarExpMsk=True,
              doBetaMsk=True, doEccMsk=True):
    self._mask = np.ones(self.x0.shape)

    if self._isROIMasked and self.doROIMsk:
        self._mask = np.all((self._mask, self._roiMsk), 0)
    if self._isVarExpMasked and self.doVarExpMsk:
        self._mask = np.all((self._mask, self._varExpMsk), 0)
    if self._isBetaMasked and self.doBetaMsk:
        self._mask = np.all((self._mask, self._betaMsk), 0)
    if self._isEccMasked and self.doEccMsk:
        self._mask = np.all((self._mask, self._eccMsk), 0)

    self._mask = self._mask.astype(bool)

    return self._mask

#----------------------------------------------------------------------------#
# return number of voxels per dartboard element
# possible values for split: 8, 8tilt, 4, 4tilt


def maskDartBoard(self, split='8'):
    ppi = 2 * np.pi
    pi2 = np.pi / 2
    pi4 = np.pi / 4
    pi8 = np.pi / 8

    if split == '8':
        nVoxDart = np.empty((18))
        k = 1

        # central
        nVoxDart[0] = np.sum(self.r <= 2)

        for i in range(2):
            for j in range(8):
                if i == 0:
                    nVoxDart[k] = np.sum(np.all([self.r > 2, self.r <= 4.5, self.phi > j * pi4, self.phi <= (j + 1) * pi4], 0))
                elif i == 1:
                    nVoxDart[k] = np.sum(np.all([self.r > 4.5, self.r <= 7, self.phi > j * pi4, self.phi <= (j + 1) * pi4], 0))
                k += 1

        # outer
        nVoxDart[17] = np.sum(self.r > 7)

    if split == '16':
        nVoxDart = np.empty((34))
        k = 1

        # central
        nVoxDart[0] = np.sum(self.r <= 2)

        for i in range(2):
            for j in range(16):
                if i == 0:
                    nVoxDart[k] = np.sum(np.all([self.r > 2, self.r <= 4.5, self.phi >= j * pi8, self.phi < (j + 1) * pi8], 0))
                elif i == 1:
                    nVoxDart[k] = np.sum(np.all([self.r > 4.5, self.r <= 7, self.phi >= j * pi8, self.phi < (j + 1) * pi8], 0))
                k += 1

        # outer
        nVoxDart[33] = np.sum(self.r > 7)

    elif split == '8tilt':
        # instead of tilting the grid -pi/8, tilt the data +pi/8
        self.phi += pi8
        self.phi[self.phi > ppi] -= ppi

        nVoxDart = np.empty((18))
        k = 1

        # central
        nVoxDart[0] = np.sum(self.r <= 2)

        for i in range(2):
            for j in range(8):
                if i == 0:
                    nVoxDart[k] = np.sum(np.all([self.r > 2, self.r <= 4.5, self.phi > j * pi4, self.phi <= (j + 1) * pi4], 0))
                elif i == 1:
                    nVoxDart[k] = np.sum(np.all([self.r > 4.5, self.r <= 7, self.phi > j * pi4, self.phi <= (j + 1) * pi4], 0))
                k += 1

        # outer
        nVoxDart[17] = np.sum(self.r > 7)

        # tilt back (reinitialize polar coords)

    elif split == '4':
        nVoxDart = np.empty((10))
        k = 1

        # central
        nVoxDart[0] = np.sum(self.r <= 2)

        for i in range(2):
            for j in range(4):
                if i == 0:
                    nVoxDart[k] = np.sum(np.all([self.r > 2, self.r <= 4.5, self.phi > j * pi2, self.phi <= (j + 1) * pi2], 0))
                elif i == 1:
                    nVoxDart[k] = np.sum(np.all([self.r > 4.5, self.r <= 7, self.phi > j * pi2, self.phi <= (j + 1) * pi2], 0))
                k += 1

        # outer
        nVoxDart[9] = np.sum(self.r > 7)

    elif split == '4tilt':
        # instead of tilting the grid -pi/4, tilt the data +pi/2
        self.phi += pi4
        self.phi[self.phi > ppi] -= ppi

        nVoxDart = np.empty((10))
        k = 1

        # central
        nVoxDart[0] = np.sum(self.r <= 2)

        for i in range(2):
            for j in range(4):
                if i == 0:
                    nVoxDart[k] = np.sum(np.all([self.r > 2, self.r <= 4.5, self.phi > j * pi2, self.phi <= (j + 1) * pi2], 0))
                elif i == 1:
                    nVoxDart[k] = np.sum(np.all([self.r > 4.5, self.r <= 7, self.phi > j * pi2, self.phi <= (j + 1) * pi2], 0))
                k += 1

        # outer
        nVoxDart[9] = np.sum(self.r > 7)

        # tilt back (reinitialize polar coords)

    return nVoxDart

# load mat file from SPM analysis for tValues, convert nii to mat first in MakeAllImages.m


def loadSPMmat(self, path):
    self.tValues = loadmat(path, squeeze_me=True)['bar']

    if self.isROIMasked:
        self.tValues = self.tValues[self.roiMsk]
    if self._isVarExpMasked:
        self.tValues = self.tValues[self.msk]
