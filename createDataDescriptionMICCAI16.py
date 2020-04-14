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

sourceDataPath = '/data/MICCAI16/'
outputDirectory = '/data/output/'

json_dict = {}
json_dict['name'] = "MICCAI16"
json_dict['file-extension'] = ".nii.gz"
json_dict['output-directory'] = outputDirectory

trainingSet = []
testingSet = []

for setName in os.listdir(sourceDataPath):
        
    setPath = os.path.join(sourceDataPath, setName)

    patientSet = trainingSet if setName == 'training' else testingSet

    # For each patient: add file paths to the corresponding set
    for patientName in os.listdir(setPath):
        
        patientPath = os.path.join(setPath, patientName)

        if os.path.isdir(patientPath):                                              # Only consider directories
            
            # Create the file paths and add them to the set
            flair = os.path.join(patientPath, 'Raw', '3DFLAIR.nii.gz')
            t1 = os.path.join(patientPath, 'Raw', '3DT1.nii.gz')
            t1gado = os.path.join(patientPath, 'Raw', '3DT1GADO.nii.gz')
            dp = os.path.join(patientPath, 'Raw', 'DP.nii.gz')
            t2 = os.path.join(patientPath, 'Raw', 'T2.nii.gz')
            mask = os.path.join(patientPath, 'Preprocessed', 'Mask_registered.nii.gz')
            label = os.path.join(patientPath, 'ManualSegmentation', 'Consensus.nii.gz')
            
            # Warning: patient output directory must be a subfolder of outputDirectory: outputDirectory/patientID/outputFiles.nii.gz
            #           it cannot be outputDirectory/patientID/TemporalPoint/outputFiles.nii.gz
            patientOutputDirectory = os.path.join(outputDirectory, patientName)

            patientSet.append({'flair': flair, 't1': t1, 't1gado': t1gado, 'dp': dp, 't2': t2, 'output': patientOutputDirectory, 'mask': mask, 'label': label, 'id': patientName })

json_dict['training'] = trainingSet
json_dict['testing'] = testingSet

# Write the json file
with open("miccaiBrainSegmentationData.json", 'w', encoding='utf-8') as f:
    json.dump(json_dict, f, ensure_ascii=False, indent=4)
