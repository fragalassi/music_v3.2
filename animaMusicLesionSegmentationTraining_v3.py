#!/usr/bin/python3
# Warning: works only on unix-like systems, not windows where "python animaMusicLesionSegmentation.py ..." has to be run

import os
import sys
import shutil
import argparse
import tempfile

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

import animaMusicLesionAdditionalPreprocessingTraining_v3 as preproc

tmpFolder = tempfile.mkdtemp()

parser = argparse.ArgumentParser(
    prog='animaMusicLesionSegmentationTraining_v3.py',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Compute MS lesion segmentation using a cascaded CNN. Uses preprocessed images from animaMSExamRegistration.py')

parser.add_argument('-f', '--flair', required=True, help='Path to the MS patient FLAIR image')
parser.add_argument('-t', '--t1', required=True, help='Path to the MS patient T1 image')
parser.add_argument('-m', '--maskImage',required=True,help='path to the MS patient brain mask image')
parser.add_argument('-o', '--outputFolder',required=True,help='path to output folder')
parser.add_argument('-c','--consensus',default=True,help='path to consensus image')
parser.add_argument('-n', '--nbThreads',required=False,type=int,help='Number of execution threads (default: 0 = all cores)',default=0)

args=parser.parse_args()

t1Image=args.t1
flairImage=args.flair
maskImage=args.maskImage
outputFolder=args.outputFolder
cImage=args.consensus
nbThreads=str(args.nbThreads)

if not(os.path.isfile(cImage)):
    print("IO Error: the file "+cImage+" doesn't exist.")
    quit()
    
if not(os.path.isfile(t1Image)):
    print("IO Error: the file "+t1Image+" doesn't exist.")
    quit()

if not(os.path.isfile(flairImage)):
    print("IO Error: the file "+flairImage+" doesn't exist.")
    quit()

if not(os.path.isfile(maskImage)):
    print("IO Error: the file "+maskImage+" doesn't exist.")
    quit()
   
tmpFolder=outputFolder

# First perform additional preprocessing
print('Starting additional preprocessing of data')
preproc.music_lesion_additional_preprocessing(animaDir, animaExtraDataDir, tmpFolder, t1Image, flairImage, cImage,
                                             maskImage, nbThreads)
             
#shutil.rmtree(tmpFolder)
