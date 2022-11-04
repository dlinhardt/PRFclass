#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 09:57:44 2022

@author: dlinhardt
"""

import numpy as np
from os import path
from scipy.io import loadmat
from glob import glob

# initiate variables


def initVariables(self):
    if self._dataFrom == 'mrVista':
        self._x0      = self._model['x0']

        if self._orientation == 'VF':
            self._y0  = - self._model['y0']  # negative for same flip as in the coverage plots
        elif self._orientation == 'MP':
            self._y0  = self._model['y0']  # same orientation as MP

        a = self._model['sigma']['minor']
        b = np.sqrt(self._model['exponent'])
        self._s0      = np.divide(a, b, out=a, where=b != 0)

        self._beta0   = self._model['beta'][:, 0]  # the model beta should be the first index, rest are trends

        with np.errstate(divide='ignore', invalid='ignore'):
            self._varexp0  = (1. - self._model['rss'] / self._model['rawrss']).squeeze()

        self._maxEcc = self._params['analysis']['fieldSize']

    elif self._dataFrom == 'docker':
        self._x0      = np.hstack([m['x0'][0][0][0] for m in self._model])

        if self._orientation == 'VF':
            self._y0  = - np.hstack([m['y0'][0][0][0] for m in self._model])  # negative for same flip as in the coverage plots
        elif self._orientation == 'MP':
            self._y0  = np.hstack([m['y0'][0][0][0] for m in self._model])  # same orientation as MP

        a = np.hstack([m['sigma'][0][0]['major'][0][0][0] for m in self._model])
        b = np.hstack([m['exponent'][0][0][0] for m in self._model])
        self._s0      = np.divide(a, b, out=a, where=b != 0)

        self._beta0   = np.hstack([m['beta'][0][0][0][:,0,0] for m in self._model])  # the model beta should be the first index, rest are trends

        with np.errstate(divide='ignore', invalid='ignore'):
            self._rss0    = np.hstack([m['rss'][0][0][0] for m in self._model])
            self._rawrss0 = np.hstack([m['rawrss'][0][0][0] for m in self._model])
            self._varexp0 = 1. - self._rss0 / self._rawrss0

        self._maxEcc = np.hstack([p['analysis']['fieldSize'][0][0][0][0] for p in self._params])
        if self._hemis == '':
            if self._maxEcc[0] != self._maxEcc[1]:
                raise Warning('maxEcc for both hemispheres is different!')
        self._maxEcc = self._maxEcc[0]

    self._isROIMasked    = None
    self._isVarExpMasked = None
    self._isBetaMasked   = None
    self._isEccMasked    = None
    self._doROIMsk       = True
    self._doVarExpMsk    = True
    self._doBetaMsk      = True
    self._doEccMsk       = True


#--------------------------ALTERNATIVE  CONSTRUCTORS--------------------------#
# alternative constructor for standard vista output

def from_mrVista(cls, study, subject, session, analysis, server='ceph', forcePath=None,
                 orientation='VF', fit='fFit-fFit-fFit'):

    if server == 'ceph':
        baseP = '/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data'
    elif server == 'sacher':
        baseP = '/sacher/melange/fmri/projects'

    if forcePath:
        mat = loadmat(forcePath[0])
        # read in the corresponding coordinates
        coords = loadmat(forcePath[1])['coords'].astype('uint16')
    elif study == 'hcpret':  # spetial treatment for study hcpret
        loadROIpossible = False
        mat = loadmat(glob(path.join(baseP, study, 'hcpret_mrVista', 'subjects', subject, analysis,
                                     'Gray', '*', f'*{fit}.mat'))[0], simplify_cells=True)
        # read in the corresponding coordinates
        coords = loadmat(path.join(baseP, study, 'subjects', subject, session, 'mrVista', analysis,
                                   'Gray', 'coords.mat'), simplify_cells=True)['coords'].astype('uint16')
    else:
        loadROIpossible = True
        mat = loadmat(glob(path.join(baseP, study, 'subjects', subject, session, 'mrVista', analysis,
                                     'Gray', '*', f'*{fit}.mat'))[0], simplify_cells=True)

        # read in the corresponding coordinates
        coords = loadmat(path.join(baseP, study, 'subjects', subject, session, 'mrVista', analysis,
                                   'Gray', 'coords.mat'), simplify_cells=True)['coords'].astype('uint16')

    return cls('mrVista', study, subject, session, mat, baseP, analysis=analysis, coords=coords,
               orientation=orientation)

#--------------------------ALTERNATIVE  CONSTRUCTORS--------------------------#
# alternative constructor for Gari Docker output


def from_docker(cls, study, subject, session, task, run, method='vista',
                analysis='01', hemi='', baseP=None, orientation='VF'):
    prfanaMe = method if method.startswith('prfanalyze-') else f'prfanalyze-{method}'
    prfanaAn = analysis if analysis.startswith('analysis-') else f'analysis-{analysis}'
    subject  = subject if subject.startswith('sub-') else f'sub-{subject}'
    session  = session if session.startswith('ses-') else f'ses-{session}'
    task     = task if task.startswith('task-') else f'task-{task}'
    run      = f'{run}' if str(run).startswith('run-') else f'run-{run}'

    if not baseP:
        baseP = '/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data'

    mat = []
    hs = ['L', 'R'] if hemi == '' else [hemi]
    for h in hs:

        resP = path.join(baseP, study, 'derivatives', prfanaMe, prfanaAn, subject, session,
                         f'{subject}_{session}_{task}_{run}_hemi-{h.upper()}_results.mat')

        if not path.isfile(resP):
            raise Warning(f'file is not existant: {resP}')

        thisMat = loadmat(resP)['results']

        mat.append(thisMat)

    return cls('docker', study, subject, session, mat, baseP, task=task, run=run,
               hemis=hemi, prfanaMe=prfanaMe, prfanaAn=prfanaAn, orientation=orientation)