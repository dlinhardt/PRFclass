#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 10:18:56 2022

@author: dlinhardt
"""
import numpy as np
import scipy.stats as st
import os
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.io import loadmat
from os import path
from copy import deepcopy
from PIL import Image
from glob import glob
import matplotlib

try:
    from surfer import Brain
    from mayavi import mlab
except ModuleNotFoundError:
    print('Package pysurfer not installed!')
    print('Install it or don\'t run plot_toSurface()')


#----------------------------------------------------------------------------#
# calculate and plot the coverage maps
def _createmask(self, shape, otherRratio=None):
    x0, y0 = shape[0] // 2, shape[1] // 2
    n = shape[0]
    if otherRratio is not None:
        r = shape[0] / 2 * otherRratio
    else:
        r = shape[0] // 2

    y, x = np.ogrid[-x0:n - x0, -y0:n - y0]
    return x * x + y * y <= r * r

# plot a coverage map


def _calcCovMap(self, method='max', force=False):

    # create the filename
    if self._dataFrom == 'mrVista':
        VEstr = f'_VarExp-{int(self._isVarExpMasked*100)}' if self._isVarExpMasked else ''
        Bstr  = f'_betaThresh-{self._isBetaMasked}' if self._isBetaMasked else ''
        Sstr  = '_MPspace' if self._orientation == 'MP' else ''
        Estr  = f'_maxEcc-{self._isEccMasked}' if self._isEccMasked else ''
        methodStr = f'_{method}'

        savePathB = path.join(self._baseP, self._study, 'plots', 'cover', 'data',
                              self.subject, self.session)
        savePathF = f'{self.subject}_{self.session}_{self._analysis}{VEstr}{Estr}{Bstr}{Sstr}{methodStr}.npy'

    elif self._dataFrom == 'docker':
        VEstr = f'-VarExp{int(self._isVarExpMasked*100)}' if self._isVarExpMasked else ''
        Bstr  = f'-betaThresh{self._isBetaMasked}' if self._isBetaMasked else ''
        Sstr  = '-MPspace' if self._orientation == 'MP' else ''
        Estr  = f'_maxEcc{self._isEccMasked}' if self._isEccMasked else ''
        hemiStr   = f'_hemi-{self._hemis.upper()}' if self._hemis != '' else ''
        methodStr = f'-{method}'

        savePathB = path.join(self._baseP, self._study, 'derivatives', 'plots', 'covmapData',
                              self.subject, self.session)
        savePathF = f'{self.subject}_{self.session}_{self._task}_{self._run}{hemiStr}_desc-{"".join(self._area)}{VEstr}{Estr}{Bstr}{Sstr}{methodStr}_covmapData.npy'

    savePath  = path.join(savePathB, savePathF)

    if path.isfile(savePath) and not force:
        self.covMap = np.load(savePath, allow_pickle=True)
        return self.covMap
    else:
        if not path.isdir(savePathB):
            os.makedirs(savePathB)

        xx = np.linspace(-self.maxEcc, self.maxEcc, int(self.maxEcc * 30))

        covMap = np.zeros((len(xx), len(xx)))

        for i in range(len(self.x)):
            kern1dx = st.norm.pdf(xx, self.x[i], self.s[i])
            kern1dy = st.norm.pdf(xx, self.y[i], self.s[i])
            kern2d  = np.outer(kern1dx, kern1dy)
            kern2d /= kern2d.max()

            if method == 'max':
                covMap = np.max((covMap, kern2d), 0)
            elif method == 'mean' or method == 'sumClip':
                covMap = np.sum((covMap, kern2d), 0)

        if method == 'mean':
            covMap /= len(self.x)

        msk = self._createmask(covMap.shape)
        covMap[~msk] = 0

        self.covMap = covMap.T

        np.save(savePath, self.covMap, allow_pickle=True)

        return self.covMap


def plot_covMap(self, method='max', cmapMin=0, title=None, show=True, save=False, force=False):
    if not show:
        matplotlib.use('PDF')
        plt.ioff()

    # create the filename
    if self._dataFrom == 'mrVista':
        VEstr = f'_VarExp-{int(self._isVarExpMasked*100)}' if self._isVarExpMasked else ''
        Bstr  = f'_betaThresh-{self._isBetaMasked}' if self._isBetaMasked else ''
        Sstr  = '_MPspace' if self._orientation == 'MP' else ''
        Estr  = f'_maxEcc-{self._isEccMasked}' if self._isEccMasked else ''
        CBstr = f'_colBar-{cmapMin}'.replace('.', '') if cmapMin != 0 else ''
        methodStr = f'_{method}'

        savePathB = path.join(self._baseP, self._study, 'plots', 'cover',
                              self.subject, self.session)
        savePathF = f'{self.subject}_{self.session}_{self._analysis}{CBstr}{VEstr}{Estr}{Bstr}{Sstr}{methodStr}.svg'

    elif self._dataFrom == 'docker':
        VEstr = f'-VarExp{int(self._isVarExpMasked*100)}' if self._isVarExpMasked else ''
        Bstr  = f'-betaThresh{self._isBetaMasked}' if self._isBetaMasked else ''
        Sstr  = '-MPspace' if self._orientation == 'MP' else ''
        Estr  = f'_maxEcc{self._isEccMasked}' if self._isEccMasked else ''
        CBstr = f'-colBar{cmapMin}'.replace('.', '') if cmapMin != 0 else ''
        hemiStr   = f'_hemi-{self._hemis.upper()}' if self._hemis != '' else ''
        methodStr = f'-{method}'

        savePathB = path.join(self._baseP, self._study, 'derivatives', 'plots', 'covMap',
                              self.subject, self.session)
        savePathF = f'{self.subject}_{self.session}_{self._task}_{self._run}{hemiStr}_desc-{"".join(self._area)}{VEstr}{Estr}{Bstr}{Sstr}{methodStr}{CBstr}_covmap.svg'

    savePath  = path.join(savePathB, savePathF)

    if not path.isdir(savePathB):
        os.makedirs(savePathB)

    if not path.isfile(savePath) or show or force:
        methods = ['max', 'mean', 'sumClip']
        if not method in methods:
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

        im = ax.imshow(self.covMap, cmap='hot', extent=(-self.maxEcc, self.maxEcc, -self.maxEcc, self.maxEcc),
                       origin='lower', vmin=cmapMin, vmax=vmax)
        ax.scatter(self.x[self.r < self.maxEcc], self.y[self.r < self.maxEcc], s=.3, c='grey')
        ax.set_xlim((-self.maxEcc, self.maxEcc))
        ax.set_ylim((-self.maxEcc, self.maxEcc))
        ax.set_aspect('equal', 'box')
        fig.colorbar(im, location='right', ax=ax)

        # draw grid
        maxEcc13 = self.maxEcc / 3
        maxEcc23 = self.maxEcc / 3 * 2
        si = np.sin(np.pi / 4) * self.maxEcc
        co = np.cos(np.pi / 4) * self.maxEcc

        for e in [maxEcc13, maxEcc23, self.maxEcc]:
            ax.add_patch(plt.Circle((0, 0), e, color='grey', fill=False, linewidth=.8))

        ax.plot((-self.maxEcc, self.maxEcc), (0, 0), color='grey', linewidth=.8)
        ax.plot((0, 0), (-self.maxEcc, self.maxEcc), color='grey', linewidth=.8)
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
        print(f'new Coverage Map saved to {savePath}')
        # plt.close('all')

    if not show:
        plt.ion()

    if save:
        return savePath
    else:
        return fig


#----------------------------------------------------------------------------#
# plot on surface
def _get_surfaceSavePath(self, param, hemi):
    VEstr = f'-VarExp{int(self._isVarExpMasked*100)}' if self._isVarExpMasked else ''
    Bstr  = f'-betaThresh{self._isBetaMasked}' if self._isBetaMasked else ''
    Pstr  = f'-{param}'

    savePathB = path.join(self._baseP, self._study, 'derivatives', 'plots', 'cortex',
                          self.subject, self.session)
    savePathF = f'{self.subject}_{self.session}_{self._task}_{self._run}_hemi-{hemi[0].upper()}_desc-{"".join(self._area)}{VEstr}{Bstr}{Pstr}_cortex'

    if not path.isdir(savePathB):
        os.makedirs(savePathB)

    return savePathB, savePathF


def _make_gif(self, frame_folder, outFilename):
    # Read the images
    frames = [Image.open(image) for image in sorted(glob(f"{frame_folder}/frame*.png"))]
    # Create the gif
    frame_one = frames[0]
    frame_one.save(
        path.join(frame_folder, outFilename),
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=500,
        loop=0,
    )
    # Delete the png-s
    [os.remove(image) for image in glob(f"{frame_folder}/frame*.png")]


def plot_toSurface(self, param='ecc', hemi='left', fmriprepAna='01', save=False,
                   forceNewPosition=False, surface='inflated', showBorders=False,
                   interactive=True, create_gif=False):

    if self._dataFrom == 'mrVista':
        print('We can not do that with non-docker data!')
    elif self._dataFrom == 'docker':

        # turn of other functionality when creating gif
        if create_gif:
            manualPosition = False
            save = False
            interactive = False
        else:
            manualPosition = True

        fsP = path.join(self._baseP, self._study, 'derivatives', 'fmriprep', f'analysis-{fmriprepAna}',
                        'sourcedata', 'freesurfer')

        pialP = path.join(fsP, self.subject, 'surf', f'{hemi[0].lower()}h.pial')
        pial  = nib.freesurfer.read_geometry(pialP)

        nVertices = len(pial[0])

        # create mask dependent on used hemisphere
        if hemi[0].upper() == 'L':
            hemiM = self._roiWhichHemi == 'L'
        elif hemi[0].upper() == 'R':
            hemiM = self._roiWhichHemi == 'R'

        roiIndFsnativeHemi = self._roiIndFsnative[hemiM]
        roiIndBoldHemi     = self._roiIndBold[hemiM]

        # write data array to plot
        plotData  = np.ones(nVertices) * np.nan

        # depending on used parameter set the plot data, colormap und ranges
        if param == 'ecc':
            plotData[roiIndFsnativeHemi] = self.r0[roiIndBoldHemi]
            cmap = 'rainbow_r'
            datMin, datMax = 0, self.maxEcc

        elif param == 'pol':
            plotData[roiIndFsnativeHemi] = self.phi0[roiIndBoldHemi]
            cmap = 'hsv'
            datMin, datMax = 0, 2 * np.pi

        elif param == 'sig':
            plotData[roiIndFsnativeHemi] = self.s0[roiIndBoldHemi]
            cmap = 'rainbow_r'
            datMin, datMax = 0, 4

        elif param == 'var':
            plotData[roiIndFsnativeHemi] = self.varexp0[roiIndBoldHemi]
            cmap = 'hot'
            datMin, datMax = 0, 1
        else:
            raise Warning('Parameter string must be in ["ecc", "pol", "sig", "var"]!')

        # set everything outside mask (ROI, VarExp, ...) to nan
        plotData = deepcopy(plotData)
        if not param == 'var':
            plotData[roiIndFsnativeHemi[~self.mask[roiIndBoldHemi]]] = np.nan

        # plot the brain
        brain = Brain(self.subject, f'{hemi[0].lower()}h', surface, subjects_dir=fsP, offscreen='auto')
        # plot the data
        brain.add_data(plotData, colormap=cmap, min=datMin, max=datMax, smoothing_steps='nearest')

        # set nan to transparent
        brain.data['surfaces'][0].module_manager.scalar_lut_manager.lut.nan_color = 0, 0, 0, 0
        brain.data['surfaces'][0].update_pipeline()

        # print borders (freesurfer)
        if showBorders:
            for ar in self._area:
                try:
                    brain.add_label('V1_exvivo.thresh', borders=True, hemi=f'{hemi[0].lower()}h',
                                    color='black', alpha=.7,
                                    subdir=path.join(fsP, self.subject, 'label'))
                except:
                    pass

        # save the positioning for left and right once per subject
        if manualPosition:
            posSavePath = path.join(self._baseP, self._study, 'derivatives', 'plots',
                                    'positioning', self.subject)
            posSaveFile = f'{self.subject}_hemi-{hemi[0].upper()}_desc-{"".join(self._area)}_cortex.npy'
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

                for iI, i in enumerate(np.linspace(-1, 89, 10)):
                    brain.show_view({'azimuth': -i, 'elevation': 90, 'distance': 350,
                                     'focalpoint': np.array([20, -130, -70])}, roll=-90)
                    brain.save_image(path.join(p, f'frame-{iI}.png'))

                self._make_gif(p, n + '.gif')

            else:
                if hemi[0].upper() == 'L':
                    brain.show_view({'azimuth': -57.5, 'elevation': 106, 'distance': 300,
                                     'focalpoint': np.array([-43, -23, -8])}, roll=-130)
                elif hemi[0].upper() == 'R':
                    brain.show_view({'azimuth': -127, 'elevation': 105, 'distance': 300,
                                     'focalpoint': np.array([-11, -93, -49])}, roll=142)

        if save:
            p, n = self._get_surfaceSavePath(param, hemi)
            brain.save_image(path.join(p, n + '.pdf'))

        if interactive:
            pass
            mlab.show(stop=True)
        else:
            mlab.clf()