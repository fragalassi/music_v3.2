import json
import os
import pathlib

# /local/EMISEP_Brain_Segmentation/new\ lesions/montpellier_20170112_07/brain/M0/flair/flair.nii.gz

sourceDataPath = r'/data/EMISEP_Brain_Segmentation/'
outputDirectory = '/data/output/'

json_dict = {}
json_dict['name'] = "EMISEP"
json_dict['file-extension'] = ".nii.gz"
json_dict['output-directory'] = outputDirectory

patients = []

for patientName in os.listdir(sourceDataPath):
    
    patientPath = os.path.join(sourceDataPath, patientName)

    if os.path.isdir(patientPath):
        
        for temporalPoint in ['M0', 'M24']:

            flair = os.path.join(sourceDataPath, patientName, 'brain', temporalPoint, 'flair', 'flair.nii.gz')
            t1 = os.path.join(sourceDataPath, patientName, 'brain', temporalPoint, 't1', 't1.nii.gz')
            t2 = os.path.join(sourceDataPath, patientName, 'brain', temporalPoint, 't2', 't2.nii.gz')
            
            mask = os.path.join(sourceDataPath, patientName, 'brain', temporalPoint, 'flair', 'flair_noskull.nii.gz')
            consensus = os.path.join(sourceDataPath, patientName, 'brain', temporalPoint, 'flair', 'flair_lesion_manual.nii.gz')

            # Warning: patient output directory must be a subfolder of outputDirectory: outputDirectory/patientID/outputFiles.nii.gz
            #           it cannot be outputDirectory/patientID/TemporalPoint/outputFiles.nii.gz
            patientOutputDirectory = os.path.join(outputDirectory, patientName + '_' + temporalPoint)

            patients.append({'flair': flair, 't1': t1, 't2t': t2, 'output': patientOutputDirectory, 'mask': mask, 'label': consensus })


json_dict['training'] = patients[:len(patients)//2]
json_dict['test'] = patients[len(patients)//2:]

with open("emisepBrainSegmentationData.json", 'w', encoding='utf-8') as f:
    json.dump(json_dict, f, ensure_ascii=False, indent=4)
