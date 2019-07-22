#!/usr/bin/python

import argparse
import sys

if sys.version_info[0] > 2:
    import configparser as ConfParser
else:
    import ConfigParser as ConfParser

import os
import shutil
import tempfile
from subprocess import call, check_output

configFilePath = os.path.expanduser("~") + "/.anima/config.txt"
if not os.path.exists(configFilePath):
    print('Please create a configuration file for Anima python scripts. Refer to the README')
    quit()

configParser = ConfParser.RawConfigParser()
configParser.read(configFilePath)

animaDir = configParser.get("anima-scripts", 'anima')
animaExtraDataDir = configParser.get("anima-scripts", 'extra-data-root')
animaScriptsDir="/udd/fgalassi/Anima-Scripts"

parser = argparse.ArgumentParser(
    prog='animaMSExamPreparation',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Registers and pre-processes input images of an MS patient sequence onto a common reference.")

parser.add_argument('-r', '--reference', required=True, help='Path to the MS patient reference image (usually FLAIR at first time point)')
parser.add_argument('-f', '--flair', required=True, help='Path to the MS patient FLAIR image to register')
parser.add_argument('-t', '--t1', required=True, help='Path to the MS patient T1 image to register')
parser.add_argument('-g', '--t1-gd', default="", help='Path to the MS patient T1-Gd image to register')
parser.add_argument('-T', '--t2', default="", help='Path to the MS patient T2 image to register')
parser.add_argument('-o','--outputFolder',required=True,help='path to output image')

args = parser.parse_args()
#tmpFolder = tempfile.mkdtemp()
tmpFolder=args.outputFolder

# Anima commands
animaPyramidalBMRegistration = os.path.join(animaDir, "animaPyramidalBMRegistration")
animaMaskImage = os.path.join(animaDir, "animaMaskImage")
animaNLMeans = os.path.join(animaDir, "animaNLMeans")
animaN4BiasCorrection = os.path.join(animaDir, "animaN4BiasCorrection")
animaConvertImage = os.path.join(animaDir, "animaConvertImage")
#animaBrainExtractionScript = os.path.join(animaScriptsDir, "brain_extraction", "animaAtlasBasedBrainExtraction.py")

refImage = args.reference
listImages = [args.flair, args.t1]
if args.t1_gd != "":
    listImages.append(args.t1_gd)
if args.t2 != "":
    listImages.append(args.t2)
    
brainExtractionCommand = ["python3", "animaAtlasBasedBrainExtraction.py", "-i", refImage, "-S"]
call(brainExtractionCommand)

print('ref image', refImage)

# Decide on whether to use large image setting or small image setting
command = [animaConvertImage, "-i", refImage, "-I"]
convert_output = check_output(command, universal_newlines=True)
size_info = convert_output.split('\n')[1].split('[')[1].split(']')[0]
large_image = False
for i in range(0, 3):
    size_tmp = int(size_info.split(', ')[i])
    if size_tmp >= 350:
        large_image = True
        break

pyramidOptions = ["-p", "4", "-l", "1"]
if large_image:
    pyramidOptions = ["-p", "5", "-l", "2"]

refImagePrefix = os.path.splitext(refImage)[0]
print(refImagePrefix)
if os.path.splitext(refImage)[1] == '.gz':
    refImagePrefix = os.path.splitext(refImagePrefix)[0]

brainMask = refImagePrefix + "_brainMask.nrrd"

print('brain Mask', brainMask)

# Main loop
for i in range(0, len(listImages)):
    inputPrefix = os.path.splitext(listImages[i])[0]
    if os.path.splitext(listImages[i])[1] == '.gz':
        inputPrefix = os.path.splitext(inputPrefix)[0]

    registeredDataFile = os.path.join(tmpFolder, "SecondImage_registered.nrrd")
    rigidRegistrationCommand = [animaPyramidalBMRegistration, "-r", refImage, "-m", listImages[i], "-o",
                                registeredDataFile] + pyramidOptions
    call(rigidRegistrationCommand)

    unbiasedSecondImage = os.path.join(tmpFolder, "SecondImage_unbiased.nrrd")
    biasCorrectionCommand = [animaN4BiasCorrection, "-i", registeredDataFile, "-o", unbiasedSecondImage, "-B", "0.3"]
    call(biasCorrectionCommand)

    nlmSecondImage = os.path.join(tmpFolder, "SecondImage_unbiased_nlm.nrrd")
    nlmCommand = [animaNLMeans, "-i", unbiasedSecondImage, "-o", nlmSecondImage, "-n", "3"]
    call(nlmCommand)

    outputPreprocessedFile = inputPrefix + "_preprocessed.nrrd"
    secondMaskCommand = [animaMaskImage, "-i", nlmSecondImage, "-m", brainMask, "-o", outputPreprocessedFile]
    call(secondMaskCommand)

#shutil.rmtree(tmpFolder)
