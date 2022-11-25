#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 10:18:56 2022

@author: dlinhardt
"""

import json
import os
from copy import deepcopy
from glob import glob
from os import path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.stats as st
from PIL import Image

try:
    from mayavi import mlab
except ModuleNotFoundError:
    print('Package mayavi not installed!')
try:
    from surfer import Brain
except ModuleNotFoundError:
    print('Package pysurfer not installed!')
    print('Install it or don\'t run plot_toSurface()')


class label():
    """
    Mini class that defines a label, needed to plot borders
    """
    def __init__(self, ar, at, he, allAreaFiles):
        self.name = ar
        self.hemi = he[0].lower()[0] + 'h'

        vJsonName = [j for j in allAreaFiles if f'hemi-{he[0].upper()}_' in path.basename(j) and
                                                f'desc-{ar}-' in path.basename(j) and
                                                at in path.basename(j)][0]
        print(vJsonName)
        with open(vJsonName, 'r') as fl:
            maskinfo = json.load(fl)

        self.vertices = np.array(maskinfo['roiIndOrig'])


#----------------------------------------------------------------------------#
def _createmask(self, shape, otherRratio=None):
    """
    This creates a round mask of given size

    Args:
        shape (tuple): Tuple giving the shape of the mask (squared)
        otherRratio (float, optional): Ratio of area to mask, otherwise full size. Defaults to None.

    Returns:
        _type_: _description_
    """

    x0, y0 = shape[0] // 2, shape[1] // 2
    n = shape[0]
    if otherRratio is not None:
        r = shape[0] / 2 * otherRratio
    else:
        r = shape[0] // 2

    y, x = np.ogrid[-x0:n - x0, -y0:n - y0]
    return x * x + y * y <= r * r


def _calcCovMap(self, method='max', force=False):
    """
    Calculates the coverage map

    Args:
        method (str, optional): Used method to calc, choose from [max, mean]. Defaults to 'max'.
        force (bool, optional): Force overwire if file already exists. Defaults to False.

    Returns:
        self.covMap: the array that describes the coverage map
    """

    # create the filename
    if self._dataFrom == 'mrVista':
        VEstr = f'_VarExp-{int(self._isVarExpMasked*100)}' if self._isVarExpMasked else ''
        Bstr  = f'_betaThresh-{self._isBetaMasked}' if self._isBetaMasked else ''
        Sstr  = '_MPspace' if self._orientation == 'MP' else ''
        Estr  = f'_maxEcc-{self._isEccMasked}' if self._isEccMasked else ''
        methodStr = f'_{method}'

        savePathB = path.join(self._baseP, self._study, 'prfresult', self._prfanaAn,
                              'cover', 'data', self.subject, self.session)
        savePathF = f'{self.subject}_{self.session}_{self._prfanaAn}{VEstr}{Estr}{Bstr}{Sstr}{methodStr}.npy'

    elif self._dataFrom == 'docker':
        VEstr = f'-VarExp{int(self._isVarExpMasked*100)}' if self._isVarExpMasked else ''
        Bstr  = f'-betaThresh{self._isBetaMasked}' if self._isBetaMasked else ''
        Sstr  = '-MPspace' if self._orientation == 'MP' else ''
        Estr  = f'_maxEcc{self._isEccMasked}' if self._isEccMasked else ''
        hemiStr   = f'_hemi-{self._hemis.upper()}' if self._hemis != '' else ''
        methodStr = f'-{method}'
        areaStr = 'multipleAreas' if len(self._area) > 10 else "".join(self._area)

        savePathB = path.join(self._baseP, self._study, 'derivatives', 'prfresult',
                              self._prfanaAn, 'covMapData', self.subject, self.session)

        savePathF = f'{self.subject}_{self.session}_{self._task}_{self._run}{hemiStr}_desc-{areaStr}{VEstr}{Estr}{Bstr}{Sstr}{methodStr}_covmapData.npy'

    savePath  = path.join(savePathB, savePathF)

    if path.isfile(savePath) and not force:
        self.covMap = np.load(savePath, allow_pickle=True)
        return self.covMap
    else:
        if not path.isdir(savePathB):
            os.makedirs(savePathB)

        xx = np.linspace(-1*self.maxEcc, self.maxEcc, int(self.maxEcc * 30))

        covMap = np.zeros((len(xx), len(xx)))

        jj = 0
        for i in range(len(self.x)):
            kern1dx = st.norm.pdf(xx, self.x[i], self.s[i])
            kern1dy = st.norm.pdf(xx, self.y[i], self.s[i])
            kern2d  = np.outer(kern1dx, kern1dy)

            if np.max(kern2d)>0:
                jj += 1

                kern2d /= np.max(kern2d)

                if method == 'max':
                    covMap = np.max((covMap, kern2d), 0)
                elif method == 'mean' or method == 'sumClip':
                    covMap = np.sum((covMap, kern2d), 0)

        if method == 'mean':
            covMap /= jj

        msk = self._createmask(covMap.shape)
        covMap[~msk] = 0

        self.covMap = covMap.T

        np.save(savePath, self.covMap, allow_pickle=True)

        return self.covMap


def plot_covMap(self, method='max', cmapMin=0, title=None, show=True, save=False, force=False):
    """
    This plots the coverage map and eventually saves it

    Args:
        method (str, optional): Used method to calc, choose from [max, mean]. Defaults to 'max'.
        cmapMin (float, optional): Define where the covMap colorbar should start on the bottom. Defaults to 0.
        title (str, optional): Set a title. Defaults to None.
        show (bool, optional): Should we show the figure as popup. Defaults to True.
        save (bool, optional): Should we save the figure to standard path. Defaults to False.
        force (bool, optional): Should we overwrite. Defaults to False.

    Returns:
        figure: if not save this is the figure handle
    """

    if not show:
        plt.ioff()

    # create the filename
    if self._dataFrom == 'mrVista':
        VEstr = f'_VarExp-{int(self._isVarExpMasked*100)}' if self._isVarExpMasked else ''
        Bstr  = f'_betaThresh-{self._isBetaMasked}' if self._isBetaMasked else ''
        Sstr  = '_MPspace' if self._orientation == 'MP' else ''
        Estr  = f'_maxEcc-{self._isEccMasked}' if self._isEccMasked else ''
        CBstr = f'_colBar-{cmapMin}'.replace('.', '') if cmapMin != 0 else ''
        methodStr = f'_{method}'

        savePathB = path.join(self._baseP, self._study, 'prfresult', self._prfanaAn,
                              'cover', self.subject, self.session)
        savePathF = f'{self.subject}_{self.session}_{self._prfanaAn}{CBstr}{VEstr}{Estr}{Bstr}{Sstr}{methodStr}.svg'

    elif self._dataFrom == 'docker':
        VEstr = f'-VarExp{int(self._isVarExpMasked*100)}' if self._isVarExpMasked else ''
        Bstr  = f'-betaThresh{self._isBetaMasked}' if self._isBetaMasked else ''
        Sstr  = '-MPspace' if self._orientation == 'MP' else ''
        Estr  = f'_maxEcc{self._isEccMasked}' if self._isEccMasked else ''
        CBstr = f'-colBar{cmapMin}'.replace('.', '') if cmapMin != 0 else ''
        hemiStr   = f'_hemi-{self._hemis.upper()}' if self._hemis != '' else ''
        methodStr = f'-{method}'
        areaStr = 'multipleAreas' if len(self._area) > 10 else "".join(self._area)

        savePathB = path.join(self._baseP, self._study, 'derivatives', 'prfresult',
                              self._prfanaAn, 'covMap', self.subject, self.session)
        savePathF = f'{self.subject}_{self.session}_{self._task}_{self._run}{hemiStr}_desc-{areaStr}{VEstr}{Estr}{Bstr}{Sstr}{methodStr}{CBstr}_covmap.svg'

    savePath  = path.join(savePathB, savePathF)

    if not path.isdir(savePathB):
        os.makedirs(savePathB)

    if not path.isfile(savePath) or show or force:
        methods = ['max', 'mean', 'sumClip']
        if method not in methods:
            raise Warning(f'Chosen method "{method}" is not a available methods {methods}.')

        # calculate the coverage map
        self._calcCovMap(method, force=force)

        # set method-specific stuff
        if method == 'max':
            if cmapMin > 1 or cmapMin < 0:
                raise Warning('Choose a cmap min between 0 and 1.')
            vmax = 1
        elif method == 'mean':
            vmax = self.covMap.max()
        elif method == 'sumClip':
            vmax = 1

        fig = plt.figure(constrained_layout=True)
        ax  = plt.gca()

        im = ax.imshow(self.covMap, cmap='hot',
                       extent=(-1*self.maxEcc, self.maxEcc, -1*self.maxEcc, self.maxEcc),
                       origin='lower', vmin=cmapMin, vmax=vmax)
        ax.scatter(self.x[self.r < self.maxEcc], self.y[self.r < self.maxEcc], s=.3, c='grey')
        ax.set_xlim((-1*self.maxEcc, self.maxEcc))
        ax.set_ylim((-1*self.maxEcc, self.maxEcc))
        ax.set_aspect('equal', 'box')
        fig.colorbar(im, location='right', ax=ax)

        # draw grid
        maxEcc13 = self.maxEcc / 3
        maxEcc23 = self.maxEcc / 3 * 2
        si = np.sin(np.pi / 4) * self.maxEcc
        co = np.cos(np.pi / 4) * self.maxEcc

        for e in [maxEcc13, maxEcc23, self.maxEcc]:
            ax.add_patch(plt.Circle((0, 0), e, color='grey', fill=False, linewidth=.8))

        ax.plot((-1*self.maxEcc, self.maxEcc), (0, 0), color='grey', linewidth=.8)
        ax.plot((0, 0), (-1*self.maxEcc, self.maxEcc), color='grey', linewidth=.8)
        ax.plot((-co, co), (-si, si), color='grey', linewidth=.8)
        ax.plot((-co, co), (si, -si), color='grey', linewidth=.8)

        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.yaxis.set_ticks([np.round(maxEcc13, 1), np.round(maxEcc23, 1), np.round(self.maxEcc, 1)])
        ax.tick_params(axis="y", direction="in", pad=-fig.get_figheight() * .39 * 96)

        plt.setp(ax.yaxis.get_majorticklabels(), va="bottom")

        ax.set_box_aspect(1)

        if title is not None:
            ax.set_title(title)

    if save and not path.isfile(savePath):
        fig.savefig(savePath, bbox_inches='tight')
        print(f'new Coverage Map saved to {savePathF}')
        # plt.close('all')

    if not show:
        # plt.ion()
        pass
    if save:
        return savePath
    else:
        return fig


#----------------------------------------------------------------------------#
def _get_surfaceSavePath(self, param, hemi, surface='cortex', plain=False):
    """
    Defines the path and filename to save the Cortex plot

    Args:
        param (str): The plotted parameter
        hemi (str): The shown hemisphere
        surface (str, optional): The used surface to plot to. Defaults to 'cortex'.

    Returns:
        savePathB: The folder we save to
        savePathF: The filename we save to, without extension
    """

    VEstr = f'-VarExp{int(self._isVarExpMasked*100)}' if self._isVarExpMasked else ''
    Bstr  = f'-betaThresh{self._isBetaMasked}' if self._isBetaMasked else ''
    Pstr  = f'-{param}'

    savePathB = path.join(self._baseP, self._study, 'derivatives', 'prfresult',
                          self._prfanaAn, 'cortex', self.subject, self.session)
    ending = surface
    areaStr = 'multipleAreas' if len(self._area) > 10 else "".join(self._area)

    if not plain:
        savePathF = f'{self.subject}_{self.session}_{self._task}_{self._run}_hemi-{hemi[0].upper()}_desc-{areaStr}{VEstr}{Bstr}{Pstr}_{ending}'
    else:
        savePathF = f'{self.subject}_{self.session}_{self._task}_{self._run}_hemi-{hemi[0].upper()}_desc{Pstr}_{ending}'

    if not path.isdir(savePathB):
        os.makedirs(savePathB)

    return savePathB, savePathF


def _make_gif(self, frameFolder, outFilename):
    """
    Reads the single frames and creates GIF from them

    Args:
        frameFolder (str): Folder containing the frames as well as output folder
        outFilename (str): file name without extension
    """

    # Read the images
    frames = [Image.open(image) for image in sorted(glob(f"{frameFolder}/frame*.png"))]
    # Create the gif
    frame_one = frames[0]
    frame_one.save(
        path.join(frameFolder, outFilename),
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=500,
        loop=0,
    )
    print(f'new Cortex Map saved to {outFilename}')
    # Delete the png-s
    [os.remove(image) for image in glob(f"{frameFolder}/frame*.png")]


def plot_toSurface(self, param='ecc', hemi='left', fmriprepAna='01', save=False,
                    forceNewPosition=False, surface='inflated',
                    showBordersAtlas=None, showBordersArea=None,
                    interactive=True, create_gif=False, headless=False):
    """
    If we have docker data that was analyzed in fsnative space we can plot
    a given parameter to the cortex of one hemisphere and create a
    screenshot or gif.

    Args:
        param (str, optional): The parameter to plot to the surface, choose from [ecc,pol,sig,var]. Defaults to 'ecc'.
        hemi (str, optional): Hemisphere to show, choose from [both,L,R]. Defaults to 'left'.
        fmriprepAna (str, optional): The analysis number of fMRIPrep, so we can find the freesurfer folder. Defaults to '01'.
        save (bool, optional): Should we save the screenshot. Defaults to False.
        forceNewPosition (bool, optional): If manual positioning was done already this forces us to define this anew. Defaults to False.
        surface (str, optional): Choose the freesurfer surface to plot on, if sphere gif and manualPosition is disabeled. Defaults to 'inflated'.
        showBordersAtlas (list, optional): Define the atlas to show the area borders from. Defaults to None.
        showBordersArea (list, optional): Define the areas to show borders. Defaults to None.
        interactive (bool, optional): Set if we should be able to interactively move the plot. Defaults to True.
        create_gif (bool, optional): Should we create a GIF, this disabels manual positioning. Defaults to False.
        headless (bool, optional): This supresses all pop-ups. Defaults to False.
    """

    if self._dataFrom == 'mrVista':
        print('We can not do that with non-docker data!')
    elif self._dataFrom == 'docker':

        if self._analysisSpace == 'volume':
            print('We can not yet do that with volumentric data!')
            return


        if headless:
            mlab.options.offscreen = True
            # mlab.init_notebook('x3d', 800, 800)
        else:
            mlab.options.offscreen = False

        if surface == 'sphere':
            create_gif     = False

        # turn of other functionality when creating gif
        if create_gif:
            manualPosition = False
            save = False
            interactive = True
        else:
            manualPosition = True if not surface == 'sphere' else False

        fsP = path.join(self._baseP, self._study, 'derivatives', 'fmriprep',
                        f'analysis-{fmriprepAna}', 'sourcedata', 'freesurfer')

        if hemi == 'both':
            hemis = ['L', 'R']
        else:
            hemis = [hemi]

        for hemi in hemis:

            if save:
                p, n = self._get_surfaceSavePath(param, hemi, surface)
                if path.isfile(path.join(p, n + '.pdf')):
                    return

            if create_gif:
                p, n = self._get_surfaceSavePath(param, hemi)
                if path.isfile(path.join(p, n + '.gif')):
                    return

            pialP = path.join(fsP, self.subject, 'surf', f'{hemi[0].lower()}h.pial')
            pial  = nib.freesurfer.read_geometry(pialP)

            nVertices = len(pial[0])

            # create mask dependent on used hemisphere
            if hemi[0].upper() == 'L':
                hemiM = self._roiWhichHemi == 'L'
            elif hemi[0].upper() == 'R':
                hemiM = self._roiWhichHemi == 'R'

            roiIndOrigHemi = self._roiIndOrig[hemiM]
            roiIndBoldHemi     = self._roiIndBold[hemiM]

            # write data array to plot
            plotData  = np.ones(nVertices) * np.nan

            # depending on used parameter set the plot data, colormap und ranges
            if param == 'ecc':
                plotData[roiIndOrigHemi] = self.r0[roiIndBoldHemi]
                cmap = 'rainbow_r'
                datMin, datMax = 0, self.maxEcc

            elif param == 'pol':
                plotData[roiIndOrigHemi] = self.phi0[roiIndBoldHemi]
                cmap = 'hsv'
                datMin, datMax = 0, 2 * np.pi

            elif param == 'sig':
                plotData[roiIndOrigHemi] = self.s0[roiIndBoldHemi]
                cmap = 'rainbow_r'
                datMin, datMax = 0, 4

            elif param == 'var':
                plotData[roiIndOrigHemi] = self.varexp0[roiIndBoldHemi]
                cmap = 'hot'
                datMin, datMax = 0, 1
            else:
                raise Warning('Parameter string must be in ["ecc", "pol", "sig", "var"]!')

            # set everything outside mask (ROI, VarExp, ...) to nan
            plotData = deepcopy(plotData)
            if not param == 'var':
                plotData[roiIndOrigHemi[~self.mask[roiIndBoldHemi]]] = np.nan

            # plot the brain
            brain = Brain(self.subject, f'{hemi[0].lower()}h', surface, subjects_dir=fsP)
            # plot the data
            brain.add_data(np.float16(plotData), colormap=cmap, min=datMin, max=datMax,
                           smoothing_steps='nearest', remove_existing=True)

            # set nan to transparent
            brain.data['surfaces'][0].module_manager.scalar_lut_manager.lut.nan_color = 0, 0, 0, 0
            brain.data['surfaces'][0].update_pipeline()

            # print borders (freesurfer)
            if showBordersArea is not None and showBordersAtlas is not None:
                if showBordersAtlas == 'all':
                    ats = self._atlas
                elif isinstance(showBordersAtlas, list):
                    ats = showBordersAtlas
                elif isinstance(showBordersAtlas, str):
                    ats = [showBordersAtlas]
                elif showBordersAtlas is True:
                    ats = ['benson']

                if isinstance(showBordersArea, list):
                    ars = showBordersArea
                else:
                    ars = self._area

                for at in ats:
                    for ar in ars:
                        try:
                            brain.add_label(label(ar, at, hemi, self._allAreaFiles),
                                            borders=True, color='black', alpha=.7)
                        except:
                            pass

            # save the positioning for left and right once per subject
            if manualPosition:
                posSavePath = path.join(self._baseP, self._study, 'derivatives', 'prfresult',
                                        'positioning', self.subject)
                areaStr = 'multipleAreas' if len(self._area) > 10 else "".join(self._area)

                posSaveFile = f'{self.subject}_hemi-{hemi[0].upper()}_desc-{areaStr}_cortex.npy'
                posPath  = path.join(posSavePath, posSaveFile)

                if not path.isdir(posSavePath):
                    os.makedirs(posSavePath)

                if not path.isfile(posPath) or forceNewPosition:
                    if hemi[0].upper() == 'L':
                        brain.show_view({'azimuth': -57.5, 'elevation': 106, 'distance': 300,
                                         'focalpoint': np.array([-43, -23, -8])}, roll=-130)
                    elif hemi[0].upper() == 'R':
                        brain.show_view({'azimuth': -127, 'elevation': 105, 'distance': 300,
                                         'focalpoint': np.array([-11, -93, -49])}, roll=142)

                    mlab.show(stop=True)
                    pos = np.array(brain.show_view(), dtype='object')
                    np.save(posPath, pos.astype('object'), allow_pickle=True)
                    # print(pos)
                else:
                    pos = np.load(posPath, allow_pickle=True)
                    # print(pos)
                    brain.show_view({'azimuth': pos[0][0], 'elevation': pos[0][1], 'distance': pos[0][2],
                                     'focalpoint': pos[0][3]}, roll=pos[1])
            else:
                if create_gif:
                    p, n = self._get_surfaceSavePath(param, hemi)

                    if hemi[0].upper() == 'L':
                        for iI, i in enumerate(np.linspace(-1, 89, 10)):
                            brain.show_view({'azimuth': -i, 'elevation': 90, 'distance': 350,
                                             'focalpoint': np.array([30, -130, -60])}, roll=-90)
                            brain.save_image(path.join(p, f'frame-{iI}.png'))

                    elif hemi[0].upper() == 'R':
                        for iI, i in enumerate(np.linspace(-1, 89, 10)):
                            brain.show_view({'azimuth': i, 'elevation': -90, 'distance': 350,
                                             'focalpoint': np.array([-30, -130, -60])}, roll=90)
                            brain.save_image(path.join(p, f'frame-{iI}.png'))

                    self._make_gif(p, n + '.gif')

                else:
                    if surface == 'sphere':
                        if hemi[0].upper() == 'L':
                            brain.show_view({'azimuth': -80, 'elevation': 125, 'distance': 500,
                                             'focalpoint': np.array([0, 0, 0])}, roll=-170)
                        elif hemi[0].upper() == 'R':
                            brain.show_view({'azimuth': 80, 'elevation': -125, 'distance': 500,
                                             'focalpoint': np.array([0, 0, 0])}, roll=170)

                    else:
                        if hemi[0].upper() == 'L':
                            brain.show_view({'azimuth': -57.5, 'elevation': 106, 'distance': 300,
                                             'focalpoint': np.array([-43, -23, -8])}, roll=-130)
                        elif hemi[0].upper() == 'R':
                            brain.show_view({'azimuth': -127, 'elevation': 105, 'distance': 300,
                                             'focalpoint': np.array([-11, -93, -49])}, roll=142)

            if save:
                p, n = self._get_surfaceSavePath(param, hemi, surface)
                brain.save_image(path.join(p, n + '.pdf'))
                print(f'new Cortex Map saved to {path.join(p, n + ".pdf")}')

            if interactive:
                mlab.show(stop=True)
            else:
                mlab.clf()
