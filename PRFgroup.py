import numpy as np
from os import path
from glob import glob
import re
import pandas as pd
from tqdm import tqdm
from . import PRF

class PRFgroup():
    """
    This is a list class for PRFclass.
    You can give study and masks for subjects/sessions/...
    as regexp or it will load everything.
    """

    def __init__(self,
                 data_from,
                 study,
                 DF,
                 baseP,
                 orientation,
                 method=None,
                 prfanalyze=None,
                 hemi=None,
                 fit=None,
                 ):

        self.data_from = data_from
        self.study = study
        self.data = DF
        self.baseP = baseP
        self.orientation = orientation

        if method:
            self._method = method
        if prfanalyze:
            self._prfanalyze = prfanalyze
        if hemi:
            self._hemi = hemi
        if fit:
            self._fit = fit

    def __getitem__(self, item):
        return self.data.loc[item]
     
#--------------------------ALTERNATIVE  CONSTRUCTORS--------------------------#
    @classmethod
    def from_docker(cls,
                    study,
                    subjects='.*',
                    sessions='.*',
                    tasks='.*',
                    runs='.*',
                    method='vista',
                    prfanalyze='01',
                    hemi='',
                    baseP=None,
                    orientation='VF',
                    ):

        if not baseP:
            baseP = '/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data'

        anas = glob(path.join(baseP, study, 'derivatives', f'prfanalyze-{method}',
                              f'analysis-{prfanalyze}', '*', '*', '*_estimates.json'))

        su = [path.basename(a).split('_')[0].split('-')[-1] for a in anas]
        se = [path.basename(a).split('_')[1].split('-')[-1] for a in anas]
        ta = [path.basename(a).split('_')[2].split('-')[-1] for a in anas]
        ru = [path.basename(a).split('_')[3].split('-')[-1] for a in anas]

        anasDF = pd.DataFrame(np.array([su, se, ta, ru]).T, columns=['subject', 'session', 'task', 'run'])

        anasDF = anasDF[(anasDF['subject'].str.match(subjects)) &
                        (anasDF['session'].str.match(sessions)) &
                        (anasDF['task'].str.match(tasks)) &
                        (anasDF['run'].str.match(runs))
                        ]

        anasDF = anasDF.drop_duplicates().reset_index(drop=True)
        anasDF = anasDF.sort_values(['subject','session','task','run'], ignore_index=True)

        for I, a in anasDF.iterrows():
            try:
                anasDF.loc[I,'prf'] = PRF.from_docker(study = study,
                                                        subject = a['subject'],
                                                        session = a['session'],
                                                        task = a['task'],
                                                        run  = a['run'],
                                                        method   = method,
                                                        analysis = prfanalyze,
                                                        hemi  = hemi,
                                                        baseP = baseP,
                                                        orientation = orientation
                                                        )
            except Exception as e:
                print(f'could not load sub-{a["subject"]}_ses-{a["session"]}_task-{a["task"]}_run-{a["run"]}!')
                print(e)
                anasDF.drop(I).reset_index(drop=True)


        return cls(data_from = 'docker',
                   study = study,
                   DF    = anasDF,
                   baseP = baseP,
                   orientation = orientation,
                   hemi  = hemi,
                   prfanalyze  = prfanalyze,
                   method = method,
                   )

#------------------------- ALTERNATIVE CONSTRUCTORS -------------------------#
    @classmethod
    def from_mrVista(cls,
                     study,
                     subjects='.*',
                     sessions='.*',
                     analysis='.*',
                     baseP=None,
                     orientation='VF',
                     fit='fFit-fFit-fFit',
                     ):

        if not baseP:
            baseP = '/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data'

        anas = glob(path.join(baseP, study, 'subjects', '*', '*', 'mrVista', '*'))

        anas = [a for a in anas if len(glob(path.join(a, 'Gray', '*', f'*{fit}.mat')))]

        su = [a.split('/')[-4] for a in anas]
        se = [a.split('/')[-3] for a in anas]
        an = [a.split('/')[-1] for a in anas]

        anasDF = pd.DataFrame(np.array([su, se, an]).T, columns=['subject', 'session', 'analysis'])

        anasDF = anasDF[(anasDF['subject'].str.match(subjects)) &
                        (anasDF['session'].str.match(sessions)) &
                        (anasDF['analysis'].str.match(analysis))
                        ]

        anasDF = anasDF.drop_duplicates().reset_index()
        anasDF = anasDF.sort_values(['subject','session','analysis'], ignore_index=True)

        for I, a in anasDF.iterrows():
            try:
                anasDF.loc[I,'prf'] = PRF.from_mrVista(study = study,
                                                            subject = a['subject'],
                                                            session = a['session'],
                                                            analysis = a['analysis'],
                                                            server   = 'ceph',
                                                            forcePath = None,
                                                            orientation = orientation,
                                                            fit = fit
                                                            )
            except:
                print(f'could not load subject: {a["subject"]}; session: {a["session"]}; analysis: {a["analysis"]}!')
                anasDF.drop(I).reset_index(drop=True)


        return cls(data_from = 'mrVista',
                   study = study,
                   DF    = anasDF,
                   baseP = baseP,
                   orientation = orientation,
                   fit = fit
                   )

#---------------------------------- MASKING ---------------------------------#
    def maskROI(self, area='V1', atlas='benson', doV123=False, forcePath=False):
        for I, a in self.data.iterrows():
            a['prf'].maskROI(area, atlas, doV123, forcePath)


    def maskVarExp(self, varExpThresh, varexp_easy=False, highThresh=None, spmPath=None):
        for I, a in self.data.iterrows():
            a['prf'].maskVarExp(varExpThresh, varexp_easy, highThresh, spmPath)


    def maskEcc(self, rad):
        for I, a in self.data.iterrows():
            a['prf'].maskEcc(rad)

    def calc_subject_mask(self):
        self._subject_mask = {}
        for s in self.subject:
            self._subject_mask[s] = np.all([a['prf'].mask for I,a in self.data.iterrows()
                                           if a['subject'] == s], 0)


#----------------------------------- QUERY ----------------------------------#
    def querty(s):
        self.sub_data = self.data.query(s)

        return self.sub_data


#--------------------------------- PROPERTIS --------------------------------#
    @property
    def subject(self):
        return self.data['subject'].unique()

    @property
    def session(self):
        return self.data['session'].unique()

    @property
    def task(self):
        if self.data_from == 'docker':
            return self.data['task'].unique()
        else:
            print('No task field in mrVista data')

    @property
    def run(self):
        if self.data_from == 'docker':
            return self.data['run'].unique()
        else:
            print('No run field in mrVista data')

    @property
    def analysis(self):
        if self.data_from == 'mrVista':
            return self.data['analysis'].unique()
        else:
            print('No analysis field in docker data')

    @property
    def x0(self):
        return np.hstack([a['prf'].x0 for I,a in self.data.iterrows()])

    @property
    def y0(self):
        return np.hstack([a['prf'].y0 for I,a in self.data.iterrows()])

    @property
    def s0(self):
        return np.hstack([a['prf'].s0 for I,a in self.data.iterrows()])

    @property
    def r0(self):
        return np.hstack([a['prf'].r0 for I,a in self.data.iterrows()])

    @property
    def ecc0(self):
        return np.hstack([a['prf'].ecc0 for I,a in self.data.iterrows()])

    @property
    def phi0(self):
        return np.hstack([a['prf'].phi0 for I,a in self.data.iterrows()])

    @property
    def pol0(self):
        return np.hstack([a['prf'].pol0 for I,a in self.data.iterrows()])

    @property
    def varexp0(self):
        return np.hstack([a['prf'].varexp0 for I,a in self.data.iterrows()])

    @property
    def varexp_easy0(self):
        return np.hstack([a['prf'].varexp_easy0 for I,a in self.data.iterrows()])

    @property
    def x(self):
        return np.hstack([a['prf'].x for I,a in self.data.iterrows()])

    @property
    def y(self):
        return np.hstack([a['prf'].y for I,a in self.data.iterrows()])

    @property
    def s(self):
        return np.hstack([a['prf'].s for I,a in self.data.iterrows()])

    @property
    def r(self):
        return np.hstack([a['prf'].r for I,a in self.data.iterrows()])

    @property
    def ecc(self):
        return np.hstack([a['prf'].ecc for I,a in self.data.iterrows()])

    @property
    def phi(self):
        return np.hstack([a['prf'].phi for I,a in self.data.iterrows()])

    @property
    def pol(self):
        return np.hstack([a['prf'].pol for I,a in self.data.iterrows()])

    @property
    def varexp(self):
        return np.hstack([a['prf'].varexp for I,a in self.data.iterrows()])

    @property
    def varexp_easy(self):
        return np.hstack([a['prf'].varexp_easy for I,a in self.data.iterrows()])
    @property
    def mask(self):
        return np.hstack([a['prf'].mask for I,a in self.data.iterrows()])

    @property
    def subject_mask(self):
        if not hasattr(self, '_subject_mask'):
            self.calc_subject_mask()
        return self._subject_mask