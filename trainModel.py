#!/usr/bin/python

import argparse
import animaMusicLesionTrainModel_v3 as trainModel

# Initialize the program parser and get the arguments
parser = argparse.ArgumentParser(
    prog='animaMusicLesionTrainModel_v3.py',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Train the cascaded CNN to segment MS Lesions.')

parser.add_argument('-i', '--inputDirectory', required=True, help='Training data: the directory containing all training subjects.')
parser.add_argument('-m', '--model', default="t1_t2_flair_ce_upsampleAnima", help='Model name.')
parser.add_argument('-f', '--flair', default="FLAIR_masked_normed_nyul_upsampleAnima.nii.gz", help='FLAIR file name.')
parser.add_argument('-t', '--t1', default="T1_masked_normed_nyul_upsampleAnima.nii.gz", help='T1 file name.')
parser.add_argument('-T', '--t2', default="T2_masked_normed_nyul_upsampleAnima.nii.gz", help='T2 file name.')
parser.add_argument('-l', '--label', default="Consensus_upsampleAnima.nii.gz", help='Label file name.')

args = parser.parse_args()

train_subjects = args.inputDirectory

modelName = args.model
t1Image = args.t1
t2Image = args.t2
flairImage = args.flair
cImage = args.label

trainModel.music_lesion_train_model(train_subjects, t1Image, t2Image, flairImage, cImage, modelName)