import numpy as np

# class for loading mrVista results

class PRF:
    """
    This class loads data from PRF mapping analysis done with a matlab
    based mrVista analysis or the prfanalyze-vistasoft docker.
    Initialize with PRF.from_mrVista or PRF.from_docker method!
    """

    # load in all the other files with functions
    from ._datastuff import initVariables, from_docker, from_mrVista
    from ._maskstuff import maskROI, maskVarExp, maskEcc, maskBetaThresh, _calcMask, maskDartBoard
    from ._loadadditionalstuff import loadStim, loadTC, loadJitter, loadRealign
    from ._calculatestuff import calcKdeDiff, calcPRFprofiles, centralScotBorder, plot_kdeDiff2d
    from ._plotstuff import plot_covMap, _calcCovMap, plot_toSurface, _createmask, _get_surfaceSavePath, _make_gif

    from_docker  = classmethod(from_docker)
    from_mrVista = classmethod(from_mrVista)

    def __init__(self, dataFrom, study, subject, session, baseP, mat=None,
                 est=None, analysis=None, task=None, run=None, area=None,
                 coords=None, niftiFolder=None, hemis=None, prfanaMe=None,
                 prfanaAn=None, orientation='VF'):


        self._dataFrom = dataFrom
        self._study    = study
        self._subject  = subject
        self._session  = session
        self._baseP    = baseP
        self._area     = 'full'

        if mat:
            self._mat      = mat
        if est:
            self._estimates = est
        if analysis:
            self._analysis = analysis
        if task:
            self._task     = task
        if run:
            self._run      = run
        if area:
            self._area     = area
        if prfanaMe:
            self._prfanaMe = prfanaMe
        if prfanaAn:
            self._prfanaAn = prfanaAn
        if niftiFolder:
            self._niftiFolder     = niftiFolder
        if coords is not None:
            self._coords   = coords
        if self._dataFrom == 'docker':
            self._hemis    = hemis

        self._orientation = orientation

        if self._dataFrom == 'mrVista':
            self._model  = mat['model']
            self._params = mat['params']
        elif self._dataFrom == 'docker':
            if hasattr(self, '_mat'):
                self._model  = [m['model'][0][0][0][0]  for m in self._mat]
                self._params = [m['params'][0][0][0][0] for m in self._mat]

        # initialize
        self.initVariables()

    @property
    def subject(self):
        return self._subject

    @property
    def session(self):
        return self._session

    @property
    def y0(self):
        return self._y0

    @property
    def x0(self):
        return self._x0

    @property
    def s0(self):
        return self._s0

    @property
    def r0(self):
        return np.sqrt(np.square(self.x0) + np.square(self.y0))

    @property
    def ecc0(self):
        return np.sqrt(np.square(self.x0) + np.square(self.y0))

    @property
    def phi0(self):
        p = np.arctan2(self.y0, self.x0)
        p[p < 0] += 2 * np.pi
        return p

    @property
    def phi0_orig(self):
        p = np.arctan2(self.y0, self.x0)
        return p

    @property
    def pol0(self):
        return self.phi0

    @property
    def pol0_orig(self):
        return self.phi0_orig

    @property
    def beta0(self):
        return self._beta0

    @property
    def varexp0(self):
        return self._varexp0

    @property
    def voxelTC0(self):
        return self._voxelTC0

    @property
    def maxEcc(self):
        return self._maxEcc

    @property
    def model(self):
        return self._model

    @property
    def params(self):
        return self._params

    @property
    def doROIMsk(self):
        return self._doROIMsk

    @doROIMsk.setter
    def doROIMsk(self, value):
        self._doROIMsk = value

    @property
    def doVarExpMsk(self):
        return self._doVarExpMsk

    @doVarExpMsk.setter
    def doVarExpMsk(self, value):
        self._doVarExpMsk = value

    @property
    def doBetaMsk(self):
        return self._doBetaMsk

    @doBetaMsk.setter
    def doBetaMsk(self, value):
        self._doBetaMsk = value

    @property
    def doEccMsk(self):
        return self._doEccMsk

    @doEccMsk.setter
    def doEccMsk(self, value):
        self._doEccMsk = value

    @property
    def mask(self):
        return self._calcMask()

    @property
    def y(self):
        return self.y0[self.mask]

    @property
    def x(self):
        return self.x0[self.mask]

    @property
    def s(self):
        return self.s0[self.mask]

    @property
    def r(self):
        return self.r0[self.mask]

    @property
    def ecc(self):
        return self.r0[self.mask]

    @property
    def phi(self):
        return self.phi0[self.mask]

    @property
    def pol(self):
        return self.phi0[self.mask]

    @property
    def beta(self):
        return self._beta0[self.mask]

    @property
    def varexp(self):
        return self._varexp0[self.mask]

    @property
    def voxelTC(self):
        return self._voxelTC0[self.mask,:]

    @property
    def meanVarExp(self):
        # if not self.isROIMasked: self.maskROI()
        return np.nanmean(self.varexp)