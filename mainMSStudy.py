import sys
import os
import json
import subprocess
import pathlib
import argparse
from subprocess import call, check_output
import configparser as ConfParser

import animaMSExamPreparation as examPreparation
import animaMusicLesionSegmentation_v3 as lesionSegmentation
import animaMusicLesionTrainModel_v3 as trainModel


configFilePath = os.path.expanduser("~") + "/.anima/config.txt"
if not os.path.exists(configFilePath) :
    print('Please create a configuration file for Anima python scripts. Refer to the README')
    quit()

configParser = ConfParser.RawConfigParser()
configParser.read(configFilePath)

animaDir = configParser.get("anima-scripts",'anima')
animaScriptsDir = configParser.get("anima-scripts",'anima-scripts-root')
animaExtraDataDir = configParser.get("anima-scripts",'extra-data-root')
sys.path.append(animaScriptsDir)

animaApplyTransformSerie = os.path.join(animaDir,"animaApplyTransformSerie")
animaImageResolutionChanger = os.path.join(animaDir,"animaImageResolutionChanger") # not 1mm isotropic
animaTransformSerieXmlGenerator = os.path.join(animaDir,"animaTransformSerieXmlGenerator")

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
parser.add_argument('-y', '--normedNyul', action='store_true', help='Use nyul normalization.')
parser.add_argument('-e', '--skipExamPreparation', action='store_true', help='Skip the exam preparation (registration, mask, NL means, N4 bias correction). Note: the exam images must be named as prepared by the examPreparation script. Use this if you already prepared your images, or name your images as if they were prepared but keep the original image names in the .json data description file.')
parser.add_argument('-p', '--skipPreprocessing', action='store_true', help='Skip the preprocessing (mask and Nyul strandardization from uspio Atlas and resampling). Same note as for the skip exam preparation option.')
parser.add_argument('-n', '--nbThreads', required=False, type=int, help='Number of execution threads (default: 0 = all cores).', default=0)
parser.add_argument('-m', '--model', default="t1_flair_1608_ce_noDenoising_noNorm_upsampleAnima_rev1", help='Model name.')
parser.add_argument('-a', '--all', action='store_true', help='Train the model with training data, then test it with testing data.')

args = parser.parse_args()

train = args.train
dataFile = args.data
skipExamPreparation = args.skipExamPreparation
skipPreprocessing = args.skipPreprocessing
# useNyulNormalization = args.normedNyul
modelName = args.model
trainAndTest = args.all
nbThreads = str(args.nbThreads)

dataName = 'training' if train else 'testing'

with open(dataFile, 'r', encoding='utf-8') as f:
    json_dict = json.load(f)

    outputDirectory = json_dict['output-directory']
    fileExtension = json_dict['file-extension']

    dataNames = ['training', 'testing'] if trainAndTest else [dataName]

    for dataName in dataNames:
        
        train = dataName == 'training'

        useT2 = True

        # For all patient in the target dataset (training set or testing set)
        for patient in json_dict[dataName]:
            
            output = patient['output']
            if not output.startswith(outputDirectory):
                output = os.path.join(outputDirectory, output)
            
            pathlib.Path(output).mkdir(parents=True, exist_ok=True) 

            print("Patient: " + output.replace(outputDirectory, '', 1))

            flair = patient['flair']
            t1 = patient['t1']
            t2 = patient['t2'] if 't2' in patient else None
            useT2 = useT2 and t2 is not None
            mask = patient['mask'] if 'mask' in patient else None       # this will be overriden if preprocessing: 
                                                                        #   the mask will be computed again from the flair
            label = patient['label'] if 'label' in patient else None

            # Preprocess the data (if necessary)
            # if not skipExamPreparation:
            #     examPreparation.process(reference=flair, flair=flair, t1=t1, t2=t2, outputFolder=output)
            
            # Resample norm images to get 1x1x1 mm3 images
            flairResampled = os.path.join(output,"FLAIR_masked_upsampleAnima.nii.gz")
            t1Resampled = os.path.join(output,"T1_masked_upsampleAnima.nii.gz")
            t2Resampled = os.path.join(output,"T2_masked_upsampleAnima.nii.gz")
            labelResampled = os.path.join(output,"Consensus_upsampleAnima.nii.gz")

            if not os.path.exists(flairResampled) or not os.path.exists(t1Resampled) or not os.path.exists(t2Resampled) or not os.path.exists(labelResampled):
                    
                command = [animaImageResolutionChanger, "-i", flair, "-o", flairResampled,
                        "-x","1","-y","1","-z","1"]
                call(command)

                command = [animaTransformSerieXmlGenerator,"-i",os.path.join(animaExtraDataDir,"id.txt"),"-o",os.path.join(output,"idn.xml")]
                call(command)
                command = [animaApplyTransformSerie, "-i", t1, "-g", flairResampled,
                        "-t",os.path.join(output,"idn.xml"),"-o",t1Resampled]
                call(command)
                if t2:
                    command = [animaApplyTransformSerie, "-i", t2, "-g", flairResampled,
                            "-t",os.path.join(output,"idn.xml"),"-o",t2Resampled]
                    call(command)
                if label:
                    command = [animaApplyTransformSerie, "-i", label, "-g", flairResampled,
                                "-t",os.path.join(output,"id.xml"),"-o",labelResampled,"-n", "nearest"]
                    call(command)

            # flairPrefix = os.path.basename(flair)[:-len(fileExtension)]
            # mask = os.path.join(output, flairPrefix + '_brainMask.nrrd') if mask else None
            # flair = os.path.join(output, flairPrefix + '_preprocessed.nrrd')
            # t1 = os.path.join(output, os.path.basename(t1)[:-len(fileExtension)] + '_preprocessed.nrrd')
            # t2 = os.path.join(output, os.path.basename(t2)[:-len(fileExtension)] + '_preprocessed.nrrd') if t2 else None
        
            print("  Process...")
            # Compute the segmentation
            lesionSegmentation.process(flair, t1, t2, mask, label, output, nbThreads, train, True, modelName)
    
        # Train the model (if necessary)
        if train:
            print("Train...")
            # nyulNorm = '' if useNyulNormalization else ''
            t1 = 'T1_masked_upsampleAnima.nii.gz'
            t2 = 'T2_masked_upsampleAnima.nii.gz' if useT2 else None
            flair = 'FLAIR_masked_upsampleAnima.nii.gz'
            consensus = "Consensus_upsampleAnima.nii.gz"
            trainingIds = [ pathlib.Path(patient['output']).name for patient in json_dict[dataName] ]
            trainModel.music_lesion_train_model(outputDirectory, t1Image=t1, t2Image=t2, flairImage=flair, cImage=consensus, modelName=modelName, trainingIds=trainingIds)
