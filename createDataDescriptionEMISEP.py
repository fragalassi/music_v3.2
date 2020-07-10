import json
import os
import pathlib

# Create a description of the data
# Outputs a json file describing where to find the files in the dataset
# The json file has the following format:
# {
#     "name": "EMISEP",
#     "file-extension": ".nii.gz",
#     "output-directory": "/data/output/",
#     "training": [
#         {
#             "output": "/data/output/rennes_20170112_36_M0",
#             "mask": "/data/EMISEP_Brain_Segmentation/rennes_20170112_36/brain/M0/flair/flair_noskull.nii.gz",
#             "label": "/data/EMISEP_Brain_Segmentation/rennes_20170112_36/brain/M0/flair/flair_lesion_manual.nii.gz",
#             "flair": "/data/EMISEP_Brain_Segmentation/rennes_20170112_36/brain/M0/flair/flair.nii.gz",
#             "t1": "/data/EMISEP_Brain_Segmentation/rennes_20170112_36/brain/M0/t1/t1.nii.gz",
#             "t2": "/data/EMISEP_Brain_Segmentation/rennes_20170112_36/brain/M0/t2/t2.nii.gz"
#         },
#         ...
#     ]
#     "testing": [ /* same format as training */ ]
# }
# The preprocessing will output preprocessed files which will be used by the core processing.
# Thus, the preprocessing output directory of each patient must be in the "output-directory" defined above,
# so that the core processing find them.

# /data/EMISEP_Brain_Segmentation/montpellier_20170112_07/brain/M0/flair/flair.nii.gz

sourceDataPath = '/local/EMISEP_Brain_Segmentation/new_lesions_same_space/'
outputDirectory = '/local/EMISEP_Brain_Segmentation/preprocessed_music/'
# labelsDirectory = '/local/EMISEP_Brain_Segmentation/preprocessed_on_t0nl/'

# trainingIds = [6, 48, 73, 67, 59, 27, 58, 24, 13, 28, 41, 42, 38, 45, 17, 1, 36, 51, 7, 20, 21, 60, 62]
# testingIds = [26, 23, 4, 49, 46, 72, 63, 64, 5, 10, 12, 16, 19, 30, 33, 34, 50, 74]

trainingIds = [7, 59, 1, 4, 6, 10, 16, 19, 28, 33, 36, 42, 45, 48, 49, 51, 58, 60, 63, 64, 67, 72, 73, 74]
testingIds = [13, 17, 38, 5, 12, 20, 21, 23, 24, 26, 27, 30, 34, 41, 46, 50, 62]

json_dict = {}
json_dict['name'] = "EMISEP"
json_dict['file-extension'] = ".nii.gz"
json_dict['output-directory'] = outputDirectory

trainingSet = []
testingSet = []

# For each patient: add file paths to the corresponding set
for patientName in os.listdir(sourceDataPath):
    
    patientPath = os.path.join(sourceDataPath, patientName)

    if os.path.isdir(patientPath):                                              # Only consider directories

        # Find to which set the patient belongs (training set or test set)
        patientId = int(patientName.split('_')[-1])
        patientSet = trainingSet if patientId in trainingIds else testingSet

        # Do the same thing for both temporal points
        for temporalPoint in ['M0', 'M24']:

            # Create the file paths and add them to the set
            flair = os.path.join(sourceDataPath, patientName, 'brain', temporalPoint, 'flair', 'flair.nii.gz' if temporalPoint == 'M0' else 'flair_on_M0.nii.gz')
            t1 = os.path.join(sourceDataPath, patientName, 'brain', temporalPoint, 't1', 't1.nii.gz')
            t2 = os.path.join(sourceDataPath, patientName, 'brain', temporalPoint, 't2', 't2.nii.gz')
            # mask = os.path.join(sourceDataPath, patientName, 'brain', temporalPoint, 'flair', 'flair_noskull.nii.gz')
            consensus = os.path.join(sourceDataPath, patientName, 'brain', temporalPoint, 'flair', 'lesions.nii.gz')

            # Warning: patient output directory must be a subfolder of outputDirectory: outputDirectory/patientID/outputFiles.nii.gz
            #           it cannot be outputDirectory/patientID/TemporalPoint/outputFiles.nii.gz
            # patientOutputDirectory = os.path.join(outputDirectory, patientName + '_' + temporalPoint)
            patientOutputDirectory = patientName + '_' + temporalPoint

            # patientSet.append({'flair': flair, 't1': t1, 't2': t2, 'output': patientOutputDirectory, 'mask': mask, 'label': consensus, 'id': patientName })
            patientSet.append({'flair': flair, 't1': t1, 't2': t2, 'output': patientOutputDirectory, 'label': consensus, 'id': patientName })

json_dict['training'] = trainingSet
json_dict['testing'] = testingSet

# Write the json file
with open("emisepBrainSegmentationData.json", 'w', encoding='utf-8') as f:
    json.dump(json_dict, f, ensure_ascii=False, indent=4)
