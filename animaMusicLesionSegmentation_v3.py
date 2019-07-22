#!/usr/bin/python3
# Warning: works only on unix-like systems, not windows where "python animaMusicLesionSegmentation.py ..." has to be run

import os
import sys
import shutil
import argparse
import tempfile
from subprocess import call

if sys.version_info[0] < 3:
    sys.exit('Python must be of version 3 or higher for this script.')

import configparser as ConfParser

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

import animaMusicLesionAdditionalPreprocessing_v3 as preproc
#import animaMusicLesionCoreProcessing_v3 as coreproc
#import animaMusicLesionPostProcessing_v3 as postproc

#tmpFolder = tempfile.mkdtemp()

parser = argparse.ArgumentParser(
    prog='animaMusicLesionSegmentation_v3.py',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Compute MS lesion segmentation using a cascaded CNN. Uses preprocessed images from animaMSExamRegistration.py')

parser.add_argument('-f', '--flair', required=True, help='Path to the MS patient FLAIR image')
parser.add_argument('-t', '--t1', required=True, help='Path to the MS patient T1 image')
parser.add_argument('-m', '--maskImage',required=True,help='path to the MS patient brain mask image')
parser.add_argument('-o', '--outputFolder',required=True,help='path to output folder')
parser.add_argument('-n', '--nbThreads',required=False,type=int,help='Number of execution threads (default: 0 = all cores)',default=0)

args=parser.parse_args()

t1Image=args.t1
flairImage=args.flair
maskImage=args.maskImage
outputFolder=args.outputFolder
nbThreads=str(args.nbThreads)

if not(os.path.isfile(t1Image)):
    print("IO Error: the file "+t1Image+" doesn't exist.")
    quit()

if not(os.path.isfile(flairImage)):
    print("IO Error: the file "+flairImage+" doesn't exist.")
    quit()

if not(os.path.isfile(maskImage)):
    print("IO Error: the file "+maskImage+" doesn't exist.")
    quit()

#tmpFolder=os.path.dirname(flairImage)
#if not(os.path.isdir(tmpFolder)) and tmpFolder != "":
#    os.makedirs(tmpFolder)
tmpFolder=outputFolder
    
# First perform additional preprocessing
print('Starting additional preprocessing of data')
preproc.music_lesion_additional_preprocessing(animaDir, animaExtraDataDir, tmpFolder, t1Image, flairImage, 
                                              maskImage, nbThreads)                                              
## Then run core process over up images
#print('Done with additional preprocessing, starting core processing of data')
#animaExtraDataDir='/temp_dd/igrida-fs1/fgalassi/MUSIC_rev2/'
#modelName = "t1_flair_1608_ce_noNorm_upsampleAnima_rev1"
#if not(os.path.isdir(os.path.join(tmpFolder, modelName))):
#    os.makedirs(os.path.join(tmpFolder, modelName))
#t1Image = os.path.join(tmpFolder, "T1_masked-upsampleAnima.nii.gz")
#flairImage = os.path.join(tmpFolder, "FLAIR_masked-upsampleAnima.nii.gz")    
#coreproc.music_lesion_core_processing(animaExtraDataDir, t1Image, flairImage, modelName, tmpFolder)
#
## Now run post-processing
#print('Done with core processing, starting post processing of data')
#
#maskImage = os.path.join(tmpFolder, "mask-er.nrrd")
#atlasWMImage = os.path.join(tmpFolder, "ATLAS-wm_masked-reg.nrrd")
#atlasGMImage = os.path.join(tmpFolder, "ATLAS-gm_masked-reg.nrrd")
#atlasCSFImage = os.path.join(tmpFolder, "ATLAS-csf_masked-reg.nrrd")
#flairImage = args.flair
#cnnImage = os.path.join(tmpFolder, modelName, modelName + "_prob_1.nii.gz")
#
#outputImage = os.path.join(tmpFolder, modelName, modelName + "_segm.nii.gz")
#postproc.music_lesion_post_processing(animaDir, animaExtraDataDir, tmpFolder, outputImage, cnnImage, flairImage,
#                                      atlasWMImage, atlasGMImage, atlasCSFImage, maskImage, nbThreads)
                                    
## Evaluate segm performance
#SEGPERF="/udd/fgalassi/dev/SegPerfAnalyzer_build/SegPerfAnalyzer-build/SegPerfAnalyzer"
#command=[SEGPERF, "-r",  os.path.join(tmpFolder, "Consensus.nii.gz"), "-i", outputImage, "-o", os.path.join(tmpFolder, modelName, modelName + "_segm_perf"),"-s", "-l"]
#call(command)
    
#shutil.rmtree(tmpFolder)
