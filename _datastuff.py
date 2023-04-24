from glob import glob
from os import path
import json
import numpy as np
from scipy.io import loadmat


def initVariables(self):
    """
    Initializes the variables loaded by the PRF.from_ amethods
    """

    if self._dataFrom == 'mrVista':
        self._x0      = self._model['x0']

        if self._orientation == 'VF':
            self._y0  = - self._model['y0']  # negative for same flip as in the coverage plots
        elif self._orientation == 'MP':
            self._y0  = self._model['y0']  # same orientation as MP

        a = self._model['sigma']['minor']
        try:
            b = np.sqrt(self._model['exponent'])
        except:
            b = np.ones(a.shape)
        self._s0      = np.divide(a, b, out=a, where=b != 0)

        self._beta0   = self._model['beta'][:, 0]  # the model beta should be the first index, rest are trends

        with np.errstate(divide='ignore', invalid='ignore'):
            self._varexp0  = (1. - self._model['rss'] / self._model['rawrss']).squeeze()

        self._maxEcc = self._params['analysis']['fieldSize']

    elif self._dataFrom == 'docker':
        self._x0      = np.array([-e['Centerx0'] for ee in self._estimates for e in ee])

        if self._orientation == 'VF':
            self._y0  = np.array([-e['Centery0'] for ee in self._estimates for e in ee]) # negative for same flip as in the coverage plots
        elif self._orientation == 'MP':
            self._y0  = np.array([e['Centery0'] for ee in self._estimates for e in ee]) # same orientation as MP

        self._s0      = np.array([e['sigmaMajor'] for ee in self._estimates for e in ee])

        self._varexp0 = np.array([e['R2'] for ee in self._estimates for e in ee])

        if hasattr(self, '_mat'):
            a = np.hstack([m['sigma'][0][0]['major'][0][0][0] for m in self._model])
            b = np.hstack([m['exponent'][0][0][0] for m in self._model])

            self._s0      = np.divide(a, b, out=a, where=b != 0)

            self._beta0   = [e['Centerx0'] for ee in self._estimates for e in ee]  # the model beta should be the first index, rest are trends

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
def from_mrVista(cls, study, subject, session, analysis, server='ceph',
                 forcePath=None, orientation='VF', fit='fFit-fFit-fFit'):
    """
    With this constructor you can load results that were analyzed
    with matlab based vistasoft pRF analysis in Vienna.

    Args:
        study (str): Study name
        subject (str): Subject name
        session (str): Session name
        analysis (str): Name of the full analysis as found in the subjects mrVista folder
        server (str, optional): Defines the server name. Defaults to 'ceph'.
        forcePath (str, optional): Set a path for the coords.mat file when not in the correct position. Defaults to None.
        orientation (str, optional): Defines if y-axis should be flipped. Use 'VF' for yes (visual field space) or 'MP' for now (retina, microperimetry space). Defaults to 'VF'.
        fit (str, optional): Changes the loaded mrVista name. Defaults to 'fFit-fFit-fFit'.

    Returns:
        cls: Instance of the PRF class with all the results
    """

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

    return cls('mrVista', study, subject, session, mat=mat, baseP=baseP, analysis=analysis, coords=coords,
               orientation=orientation)


#--------------------------ALTERNATIVE  CONSTRUCTORS--------------------------#
def from_docker(cls, study, subject, session, task, run, method='vista',
                analysis='01', hemi='', baseP=None, orientation='VF'):
    """
    With this constructor you can load results that were analyzed
    with dockerized solution published in Lerma-Usabiaga 2020 and available
    at github.com/vistalab/PRFmodel

    Args:
        study (str): Study name (e.g. 'stimsim')
        subject (str): Subject name (e.g. 'p001')
        session (str): Session name (e.g. '001')
        task (str): Task name (e.g. 'prf')
        run (str): Run number (e.g. '01')
        method (str, optional): Different methods from the prfanalyze docker possibler, I think. Defaults to 'vista'.
        analysis (str, optional): Number of the analysis within derivatives/prfanalyze. Defaults to '01'.
        hemi (str, optional): When you want to load only one hemisphere ('L' or 'R'), empty when both. Defaults to ''.
        forcePath (str, optional): If the analysis can be found in another location than standard (/z/fmri/data/)this changes the base path. Defaults to None.
        orientation (str, optional): Defines if y-axis should be flipped. Use 'VF' for yes (visual field space) or 'MP' for now (retina, microperimetry space). Defaults to 'VF'.

    Returns:
        cls: Instance of the PRF class with all the results
    """

    prfanaMe = method if method.startswith('prfanalyze-') else f'prfanalyze-{method}'
    prfanaAn = analysis if analysis.startswith('analysis-') else f'analysis-{analysis}'
    subject  = subject if subject.startswith('sub-') else f'sub-{subject}'
    session  = session if session.startswith('ses-') else f'ses-{session}'
    task     = task if task.startswith('task-') else f'task-{task}'
    run      = f'{run}' if str(run).startswith('run-') else f'run-{run}'

    if not baseP:
        baseP = '/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data'

    mat = [] if method == 'vista' else None
    est = []
    hs = ['L', 'R'] if hemi == '' else [hemi]
    for h in hs:

        estimates = path.join(baseP, study, 'derivatives', prfanaMe, prfanaAn, subject, session,
                            f'{subject}_{session}_{task}_{run}_hemi-{h}_estimates.json')

        with open(estimates, 'r') as fl:
            this_est = json.load(fl)

        est.append(this_est)

        if method == 'vista':
            resP = path.join(baseP, study, 'derivatives', prfanaMe, prfanaAn, subject, session,
                            f'{subject}_{session}_{task}_{run}_hemi-{h}_results.mat')

            if not path.isfile(resP):
                raise Warning(f'file is not existent: {resP}')

            thisMat = loadmat(resP)['results']

            mat.append(thisMat)

    return cls('docker', study, subject, session, baseP, task=task, run=run,
               hemis=hemi, prfanaMe=prfanaMe, prfanaAn=prfanaAn, orientation=orientation,
               mat=mat, est=est)
