import sys
import os
import json
import subprocess
import pathlib
import argparse

# Find lesions or train the model with the given dataset
#
# usage: main.py [-h] -d DATA [-t TRAIN] [-p PREPROCESS] [-n NBTHREADS]

# Compute MS lesion segmentation using a cascaded CNN.

# optional arguments:
#   -h, --help            show this help message and exit
#   -d DATA, --data DATA  Path to the data description file (.json). See createDataDescription.py to create the description file.
#   -t TRAIN, --train TRAIN
#                         Train the model
#   -p PREPROCESS, --preprocess PREPROCESS
#                         Preprocess the data with anima tools.
#   -n NBTHREADS, --nbThreads NBTHREADS
#                         Number of execution threads (default: 0 = all cores)

parser = argparse.ArgumentParser(
    prog='main.py',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Compute MS lesion segmentation using a cascaded CNN.')


parser.add_argument('-d', '--data', required=True, help='Path to the data description file (.json). See createDataDescription.py to create the description file.')
parser.add_argument('-t', '--train', required=False, default=False, type=bool, help='Train the model.')

parser.add_argument('-p', '--preprocess', required=False, default=True, type=bool, help='Preprocess the data with anima tools.')
parser.add_argument('-n', '--nbThreads', required=False, type=int, help='Number of execution threads (default: 0 = all cores).', default=0)

args = parser.parse_args()

train = args.train
dataFile = args.data
preprocessData = args.preprocess
nbThreads = str(args.nbThreads)

dataName = 'training' if train else 'testing'

with open(dataFile, 'r', encoding='utf-8') as f:
    json_dict = json.load(f)

    fileExtension = json_dict['file-extension']

    # For all patient in the target dataset (training set or testing set)
    for patient in json_dict[dataName]:

        flair = patient['flair']
        t1 = patient['t1']
        t2 = patient['t2']

        output = patient['output']
        pathlib.Path(output).mkdir(parents=True, exist_ok=True) 

        # Preprocess the data (if necessary)
        if preprocessData:
            preprocessingCommand = "python3 animaMSExamPreparation.py -r " + flair + " -f " + flair + " --t1 " + t1 + " --t2 " + t2 + " -o " + output
            preprocessingProcess = subprocess.Popen(preprocessingCommand, stdout=subprocess.PIPE, shell=True)
            preprocessingProcessStatus = preprocessingProcess.wait()

            flair = os.path.join(output, os.path.basename(flair)[:-len(fileExtension)] + fileExtension)
            t1 = os.path.join(output, os.path.basename(t1)[:-len(fileExtension)] + fileExtension)
            t2 = os.path.join(output, os.path.basename(t2)[:-len(fileExtension)] + fileExtension)
        
        mask = patient['mask']
        label = patient['label']

        # Compute the segmentation
        additionalParameters = " -p -c " + label if train else ""
        segmentationCommand = "python3 animaMusicLesionSegmentation_v3.py -f " + flair + " --t1 " + t1 + " --t2 " + t2 + " -m " + mask + " -n " + nbThreads + additionalParameters + " -o " + output
        segmentationProcess = subprocess.Popen(segmentationCommand, stdout=subprocess.PIPE, shell=True)
        segmentationProcessStatus = segmentationProcess.wait()

    # Train the model (if necessary)
    if train:
        trainDirectory = json_dict['output-directory']

        trainingCommand = "PTHONHASHSEED=0 python3 animaMusicLesionTrainModel_v3.py -i " + trainDirectory
        trainingProcess = subprocess.Popen(trainingCommand, stdout=subprocess.PIPE, shell=True)
        trainingProcessStatus = trainingProcess.wait()