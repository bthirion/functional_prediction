"""
Script for RS-fMRI to task-fMRI analysis.
This one just prepares data for further analysis

Works on IBC 3mm data
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Memory, Parallel, delayed
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_prob_atlas

import ibc_public
import ibc_public.utils_data
from utils import (
    make_dictionary, adapt_components, make_parcellations,
    predict_Y_multiparcel, permuted_score, fit_regressions_imgs)

# cache
cache = '/neurospin/tmp/bthirion/rsfmri2tfmri_functional'
# this is where the cache will be put
write_dir = cache
# does not need to be the same directory as cache
# it is for the moment assumed that the data have been put there
# But it does not need top be the case

if not os.path.exists(write_dir):
    os.mkdir(write_dir)

memory = Memory(cache, verbose=0)
n_jobs = 5
n_parcellations = 20
n_parcels = 256

# mask of grey matter
_package_directory = os.path.dirname(
    os.path.abspath(ibc_public.utils_data.__file__))
mask_gm = os.path.join(
    _package_directory, '../ibc_data', 'gm_mask_3mm.nii.gz')
masker = NiftiMasker(mask_img=mask_gm, smoothing_fwhm=4, memory=cache,
                     standardize=True).fit()
dummy_masker = NiftiMasker(mask_img=mask_gm, memory=cache).fit()

"""
DERIVATIVES = '/neurospin/ibc/3mm'
from ibc_public.utils_data import (
    data_parser, SUBJECTS, LABELS, all_contrasts, data_parser, copy_db)

###############################################################################
# Get and fetch data: RS-fMRI and T-fMRI

# resting-state fMRI
wc = os.path.join(
    DERIVATIVES, 'sub-*', 'ses-*', 'func', 'wrdc*RestingState*.nii.gz')
rs_fmri = sorted(glob.glob(wc))

subjects = []
for img in rs_fmri:
    subject = img.split('/')[-4]
    subjects.append(subject)

rs_fmri_db = pd.DataFrame({'subject': subjects,
                           'path': rs_fmri})

# task fmri
mem = Memory(cachedir=cache, verbose=0)
subjects = rs_fmri_db.subject.unique()

# remove sub-08 dut ot missing data issues
subjects = [subject for subject in subjects if subject != 'sub-08']
n_subjects = len(subjects)

###############################################################################
# Access to task data
task_list = ['archi_standard', 'archi_spatial', 'archi_social',
             'archi_emotional', 'hcp_emotion', 'hcp_gambling', 'hcp_motor',
             'hcp_language', 'hcp_relational', 'hcp_social', 'hcp_wm',
             'rsvp_language', 'language']
task_list = ['archi_emotional', 'archi_social', 'archi_spatial',
             'archi_standard', 'hcp_emotion', 'hcp_gambling',
             'hcp_language', 'hcp_motor', 'hcp_relational',
             'hcp_social', 'hcp_wm',
             'rsvp_language']
task_list += ['preference', 'preference_faces', 'preference_houses',
              'preference_food', 'preference_paintings']
task_list += ['lyon_mcse', 'lyon_moto', 'lyon_visu', 'lyon_audi',
              'lyon_mveb', 'lyon_mvis', 'lyon_lec1', 'lyon_lec2',
              'bang', 'self', 'wedge', 'ring']
task_list += ['mtt_sn', 'mtt_we', 'retinotopy', 'theory_of_mind',
              'pain_movie', 'emotional_pain', 'enumeration', 'vstm']

db = data_parser(derivatives=DERIVATIVES, subject_list=subjects,
                 conditions=all_contrasts, task_list=task_list)
#df = db[db.task.isin(task_list)]
df = db.sort_values(by=['subject', 'task', 'contrast'])
df = df[df.acquisition == 'ffx']
contrasts = df.contrast.unique()

db = copy_db(df, write_dir)
db.to_csv(os.path.join(write_dir, 'contrast_imgs.csv'))

###############################################################################
# Dictionary learning of RS-fMRI

n_components = 100

make_dictionary = mem.cache(make_dictionary)

dictlearning_components_img, Y = make_dictionary(
    rs_fmri, n_components, cache, mask_gm)

dictlearning_components_img.to_filename(
    os.path.join(write_dir, 'components.nii.gz'))

# visualize the results
plot_prob_atlas(dictlearning_components_img,
                title='All DictLearning components')

###############################################################################
# Dual regression to get individual components

n_dim = 200

individual_components = Parallel(n_jobs=n_jobs)(delayed(adapt_components)(
    Y, subject, rs_fmri_db, masker, n_dim) for subject in subjects)

# visualize the results
for i, subject in enumerate(subjects):
    individual_components_img = masker.inverse_transform(
        individual_components[i])
    individual_components_img.to_filename(
        os.path.join(write_dir, 'components_%s.nii.gz' % subject))

###############################################################################
# Generate brain parcellations
from nilearn.regions import Parcellations

ward = Parcellations(method='ward', n_parcels=n_parcels,
                     standardize=False, smoothing_fwhm=4.,
                     memory='nilearn_cache', memory_level=1,
                     verbose=1, mask=mask_gm)

make_parcellations = memory.cache(make_parcellations)

parcellations = make_parcellations(ward, rs_fmri, n_parcellations, n_jobs)

for b, parcellation in enumerate(parcellations):
    parcellation.to_filename(
        os.path.join(write_dir, 'parcellation_%02d.nii.gz' % b))
"""

###############################################################################
# gather the necessary datasets

# data descriptor
df = pd.read_csv(os.path.join(write_dir, 'contrast_imgs.csv'))
subjects = df.subject.unique()
n_subjects = len(subjects)
contrasts = df.contrast.unique()
n_contrasts = len(contrasts)

data = np.array([masker.transform([
    os.path.join(write_dir, df[df.subject == subject][
        df.contrast == contrast].path.values[-1])
    for contrast in contrasts])
    for subject in subjects])

# Parcellations
parcellations = [
        os.path.join(write_dir, 'parcellation_%02d.nii.gz' % b)
        for b in range(n_parcellations)]

# Individual topographies defined from resting-RandomState
individual_components_imgs = [
    os.path.join(write_dir, 'components_%s.nii.gz' % subject)
    for subject in subjects]

###############################################################################
# Cross-validated predictions
clf = RidgeCV()
# training the model
models = Parallel(n_jobs=n_jobs)(delayed(fit_regressions_imgs)(
    individual_components_imgs, data, parcellations,
    dummy_masker, clf, i) for i in range(n_subjects))


n_splits = 5
cv = KFold(n_splits=n_splits)

scores = []
vox_scores = []
con_scores = []
permuted_con_scores = []
permuted_vox_scores = []
n_permutations = 0

for train_index, test_index in cv.split(range(n_subjects)):
    # construct the predicted maps
    for j in test_index:
        X = dummy_masker.transform(individual_components_imgs[j])
        Y = data[j]
        Y_baseline = np.mean(Y[train_index], 0)
        Y_pred = predict_Y_multiparcel(
            parcellations, dummy_masker, train_index,
            n_parcels, Y, X, models, n_jobs)
        score = 1 - (Y - Y_pred) ** 2 / Y ** 2
        vox_score_ = r2_score(Y, Y_pred, multioutput='raw_values')
        vox_score = 1 - np.sum((Y - Y_pred) ** 2, 0) / np.sum((
            Y - Y_baseline.mean(0)) ** 2, 0)
        con_score_ = r2_score(Y.T, Y_pred.T, multioutput='raw_values')
        con_score = 1 - np.sum((Y.T - Y_pred.T) ** 2, 0) / np.sum(
            (Y.T - Y_baseline.T.mean(0)) ** 2, 0)

        scores.append(score)
        vox_scores.append(vox_score)
        con_scores.append(con_score)
        if n_permutations > 0:
            permuted_con_score = permuted_score(
                Y, Y_pred, Y_baseline, n_permutations=100, seed=1)
            permuted_con_scores.append(permuted_con_score)
            # permuted_vox_scores.append(permuted_vox_score)

mean_scores = np.array(scores).mean(0)
masker.inverse_transform(mean_scores).to_filename(
    os.path.join(write_dir, 'scores.nii.gz'))

mean_mean_scores = mean_scores.mean(0)
masker.inverse_transform(mean_mean_scores).to_filename(
    os.path.join(write_dir, 'mean_score.nii.gz'))

con_scores = np.array(con_scores)

mean_vox_scores = np.array(vox_scores).mean(0)
masker.inverse_transform(mean_vox_scores).to_filename(
    os.path.join(write_dir, 'mean_vox_score.nii.gz'))

if n_permutations > 0:
    permuted_con_scores = np.array(permuted_con_scores).mean(0)
    con_percentile = np.percentile(permuted_con_scores.max(0), 95)
    print(con_percentile, np.median(permuted_con_scores.max(0)),
          permuted_con_scores.max(0))

###############################################################################
#  Ouputs, figures

from nilearn.plotting import view_img
from nilearn.plotting import plot_img_on_surf

mean_vox_scores_img = masker.inverse_transform(mean_vox_scores)
view_img(mean_vox_scores_img,
         vmax=.5,
         title='mean voxel scores').open_in_browser()

print(n_parcellations, np.mean(con_scores))

output_file = os.path.join(write_dir, 'montage_ibc.svg')
plot_img_on_surf(mean_vox_scores_img,
                 views=['lateral', 'medial'],
                 hemispheres=['left', 'right'],
                 colorbar=True, output_file=output_file)


plt.show(block=False)
