# %% test script for PRFclass in Graz
# import the class
import sys

p = "/z/fmrilab/home/dlinhardt/pythonclass"  # this is the path where you clone the class
if not p in sys.path:
    sys.path.append(p)

from PRFclass import PRF, PRFgroup
import numpy as np
from matplotlib import pyplot as plt

# define some parameters
path = "/z/fmri/data/"  # path to the study folder
study = "hcpret"  # name of the study folder

"""
remember, the data structure is as follows:
/z/fmri/data/test_study/
└── derivatives
    ├── prfanalyze-gem
    │   └── analysis-01Graz
    └── prfprepare
        └── analysis-01
"""

# %% load a single subject/condition
sub = "995174"
ses = "001"
task = "bars1bars2"
run = "01"

# load the data
ana = PRF.from_docker(
    baseP=path,
    study=study,
    subject=sub,
    session=ses,
    task=task,
    run=run,
    method="gem",
    analysis="01Graz",
    hemi="",  # here we load both hemispheres, alternatively use "L" or "R"
)

# mask the data for variance explained
ana.maskVarExp(0.01)  # variance explained threshold in percent

# mask the data for a specific ROI, those are the two options:
ana.maskROI("claus", "claus")
# ana.maskROI("V1", 'wang')

# now you can access the data, e.g. the x-position
plt.hist(ana.x, bins=np.linspace(0, 5, 26))

"""
Alternatively you also have options like:
y, r (eccentricity), pol (polar angle), varexp (variance explained), s (pRF size)

When you access them like this (ana.r), you get the masked voxels only.
If you want to access all voxels, without mask add a 0, e.g. ana.r0
"""

# %% load a group of subjects
# load the data, this will load all data from all subjects
anas = PRFgroup.from_docker(
    baseP=path,
    study=study,
    subjects=".*",  # those are regular expressions, so ".*" means everything
    sessions=".*",
    tasks=".*",
    runs=".*",
    method="gem",
    prfanalyze="01Graz",
    hemi="",  # same as above, you can also choose just one hemisphere
)

# again as before, you can mask the data
anas.maskVarExp(0.01)
anas.maskROI("claus", "claus")

# access a dataframe holding all info
# the ["prf"] field are instances of the original PRF class
anas.data

# and access the data as list of all subjects
plt.hist(anas.x, bins=np.linspace(0, 5, 26))

# or as dictionary, separated for each subject
anas.sub_x
