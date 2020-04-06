import sys
import os
import json
import subprocess
import pathlib
import argparse

import animaMSExamPreparation as examPreparation
import animaMusicLesionSegmentation_v3 as lesionSegmentation
import animaMusicLesionTrainModel_v3 as trainModel

# Find lesions or train the model with the given dataset
#
# usage: main.py [-h] -d DATA [-t TRAIN] [-e] [-p] [-n NBTHREADS]

# Compute MS lesion segmentation using a cascaded CNN.

# optional arguments:
#   -h, --help            show this help message and exit
#   -d DATA, --data DATA  Path to the data description file (.json), see createDataDescription.py to create the description file.
#   -t TRAIN, --train TRAIN
#                         Train the model.
#   -e, --skip-exam-preparation
#                         Skip the exam preparation (registration, mask, NL means, N4 bias correction).
#   -p, --skip-preprocessing
#                         Skip the preprocessing (mask and Nyul strandardization from uspio Atlas and resampling).
#   -n NBTHREADS, --nbThreads NBTHREADS
#                         Number of execution threads (default: 0 = all cores).

parser = argparse.ArgumentParser(
    prog='main.py',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Compute MS lesion segmentation using a cascaded CNN.')

parser.add_argument('-d', '--data', required=True, help='Path to the data description file (.json), see createDataDescription.py to create the description file.')
parser.add_argument('-t', '--train', action='store_true', help='Train the model.')
parser.add_argument('-e', '--skipExamPreparation', action='store_true', help='Skip the exam preparation (registration, mask, NL means, N4 bias correction). Note: the exam images must be named as prepared by the examPreparation script. Use this if you already prepared your images, or name your images as if they were prepared but keep the original image names in the .json data description file.')
parser.add_argument('-p', '--skipPreprocessing', action='store_true', help='Skip the preprocessing (mask and Nyul strandardization from uspio Atlas and resampling). Same note as for the skip exam preparation option.')
parser.add_argument('-n', '--nbThreads', required=False, type=int, help='Number of execution threads (default: 0 = all cores).', default=0)
parser.add_argument('-m', '--model', default="t1_t2_flair_ce_upsampleAnima", help='Model name.')

args = parser.parse_args()

train = args.train
dataFile = args.data
skipExamPreparation = args.skipExamPreparation
skipPreprocessing = args.skipPreprocessing
modelName = args.model
nbThreads = str(args.nbThreads)

dataName = 'training' if train else 'testing'

with open(dataFile, 'r', encoding='utf-8') as f:
    json_dict = json.load(f)

    outputDirectory = json_dict['output-directory']
    fileExtension = json_dict['file-extension']

    # For all patient in the target dataset (training set or testing set)
    for patient in json_dict[dataName]:

        output = patient['output']
        pathlib.Path(output).mkdir(parents=True, exist_ok=True) 

        print("Patient: " + output.replace(outputDirectory, '', 1))

        flair = patient['flair']
        t1 = patient['t1']
        t2 = patient['t2']
        mask = patient['mask']      # this will be overriden if preprocessing: 
                                    #   the mask will be computed again from the flair
        label = patient['label']

        # Preprocess the data (if necessary)
        if not skipExamPreparation:
            examPreparation.process(reference=flair, flair=flair, t1=t1, t2=t2, outputFolder=output)
            
        flairPrefix = os.path.basename(flair)[:-len(fileExtension)]
        mask = os.path.join(output, flairPrefix + '_brainMask.nrrd')
        flair = os.path.join(output, flairPrefix + '_preprocessed.nrrd')
        t1 = os.path.join(output, os.path.basename(t1)[:-len(fileExtension)] + '_preprocessed.nrrd')
        t2 = os.path.join(output, os.path.basename(t2)[:-len(fileExtension)] + '_preprocessed.nrrd')
    
        print("  Process...")
        # Compute the segmentation
        lesionSegmentation.process(flair, t1, t2, mask, label, output, nbThreads, train, skipPreprocessing, modelName)

    # Train the model (if necessary)
    if train:
        print("Train...")
        trainModel.music_lesion_train_model(outputDirectory, t1Image="T1_masked_normed_nyul_upsampleAnima.nii.gz", t2Image="T2_masked_normed_nyul_upsampleAnima.nii.gz", flairImage="FLAIR_masked_normed_nyul_upsampleAnima.nii.gz", cImage="Consensus_upsampleAnima.nii.gz", modelName=modelName)
