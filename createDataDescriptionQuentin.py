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

# ~/quentin/su_MS001/structural$ ls
# COGNISEP_001_003_3D_FLAIR.nii.gz  T13D_MPRAGE_MS001.nii.gz

sourceDataPath = '/home/amasson/quentin/'
outputDirectory = '/data/output/'

json_dict = {}
json_dict['name'] = "Quentin"
json_dict['file-extension'] = ".nii.gz"
json_dict['output-directory'] = outputDirectory

trainingSet = []
testingSet = []

for patientName in os.listdir(sourceDataPath):

    patientPath = os.path.join(sourceDataPath, patientName, 'structural')

    if not patientName.startswith('su_MS') or not os.path.isdir(patientPath):
        continue
    
    patientID = patientName.replace('su_MS', '')

    flair = os.path.join(patientPath, 'COGNISEP_' + patientID + '_003_3D_FLAIR.nii.gz')
    t1 = os.path.join(patientPath, 'T13D_MPRAGE_MS' + patientID + '.nii.gz')
    
    # Warning: patient output directory must be a subfolder of outputDirectory: outputDirectory/patientID/outputFiles.nii.gz
    #           it cannot be outputDirectory/patientID/TemporalPoint/outputFiles.nii.gz
    patientOutputDirectory = patientName

    testingSet.append({'flair': flair, 't1': t1, 'output': patientOutputDirectory, 'id': patientName })

json_dict['training'] = trainingSet
json_dict['testing'] = testingSet

# Write the json file
with open("quentinBrainSegmentationData.json", 'w', encoding='utf-8') as f:
    json.dump(json_dict, f, ensure_ascii=False, indent=4)
