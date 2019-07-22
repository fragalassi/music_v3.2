#!/usr/bin/python
# Warning: works only on unix-like systems, not windows where "python animaAtlasBasedBrainExtraction.py ..." has to be run

import sys
import argparse

if sys.version_info[0] > 2:
    import configparser as ConfParser
else:
    import ConfigParser as ConfParser

import glob
import os
from shutil import copyfile
from subprocess import call, check_output

configFilePath = os.path.expanduser("~") + "/.anima/config.txt"
if not os.path.exists(configFilePath):
    print('Please create a configuration file for Anima python scripts. Refer to the README')
    quit()

configParser = ConfParser.RawConfigParser()
configParser.read(configFilePath)

animaDir = configParser.get("anima-scripts", 'anima')
animaExtraDataDir = configParser.get("anima-scripts", 'extra-data-root')
animaPyramidalBMRegistration = os.path.join(animaDir, "animaPyramidalBMRegistration")
animaDenseSVFBMRegistration = os.path.join(animaDir, "animaDenseSVFBMRegistration")
animaTransformSerieXmlGenerator = os.path.join(animaDir, "animaTransformSerieXmlGenerator")
animaApplyTransformSerie = os.path.join(animaDir, "animaApplyTransformSerie")
animaConvertImage = os.path.join(animaDir, "animaConvertImage")
animaMaskImage = os.path.join(animaDir, "animaMaskImage")

# Argument parsing
parser = argparse.ArgumentParser(
    description="Computes the brain mask of images given in input by registering a known atlas on it. Their output is prefix_brainMask.nrrd and prefix_masked.nrrd")

parser.add_argument('-S', '--second-step', action='store_true',
                    help="Perform second step of atlas based cropping (might crop part of the external part of the brain)")

parser.add_argument('-i', '--input', type=str, required=True, help='DWI file to process')

args = parser.parse_args()

numImages = len(sys.argv) - 1
atlasImage = animaExtraDataDir + "icc_atlas/Reference_T1.nrrd"
atlasImageMasked = animaExtraDataDir + "icc_atlas/Reference_T1_masked.nrrd"
iccImage = animaExtraDataDir + "icc_atlas/BrainMask.nrrd"

brainImage = args.input
print("Brain masking image: " + brainImage)

# Get floating image prefix
brainImagePrefix = os.path.splitext(brainImage)[0]
if os.path.splitext(brainImage)[1] == '.gz':
    brainImagePrefix = os.path.splitext(brainImagePrefix)[0]

# Decide on whether to use large image setting or small image setting
command = [animaConvertImage, "-i", brainImage, "-I"]
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

# Rough mask with whole brain
command = [animaPyramidalBMRegistration, "-m", atlasImage, "-r", brainImage, "-o", brainImagePrefix + "_rig.nrrd",
           "-O", brainImagePrefix + "_rig_tr.txt", "--sp", "3"] + pyramidOptions
call(command)

command = [animaPyramidalBMRegistration, "-m", atlasImage, "-r", brainImage, "-o", brainImagePrefix + "_aff.nrrd",
           "-O", brainImagePrefix + "_aff_tr.txt", "-i", brainImagePrefix + "_rig_tr.txt", "--sp", "3", "--ot",
           "2"] + pyramidOptions
call(command)

command = [animaDenseSVFBMRegistration, "-r", brainImage, "-m", brainImagePrefix + "_aff.nrrd", "-o",
           brainImagePrefix + "_nl.nrrd", "-O", brainImagePrefix + "_nl_tr.nrrd", "--sr", "1"] + pyramidOptions
call(command)

command = [animaTransformSerieXmlGenerator, "-i", brainImagePrefix + "_aff_tr.txt", "-i",
           brainImagePrefix + "_nl_tr.nrrd", "-o", brainImagePrefix + "_nl_tr.xml"]
call(command)

command = [animaApplyTransformSerie, "-i", iccImage, "-t", brainImagePrefix + "_nl_tr.xml", "-g", brainImage, "-o",
           brainImagePrefix + "_rough_brainMask.nrrd", "-n", "nearest"]
call(command)

command = [animaMaskImage, "-i", brainImage, "-m", brainImagePrefix + "_rough_brainMask.nrrd", "-o",
           brainImagePrefix + "_rough_masked.nrrd"]
call(command)

brainImageRoughMasked = brainImagePrefix + "_rough_masked.nrrd"

if args.second_step is True:
    print("Fine mask with masked brain...")
    # Fine mask with masked brain
    command = [animaPyramidalBMRegistration, "-m", atlasImageMasked, "-r", brainImageRoughMasked, "-o",
               brainImagePrefix + "_rig.nrrd", "-O", brainImagePrefix + "_rig_tr.txt", "--sp", "3"] + pyramidOptions
    call(command)

    command = [animaPyramidalBMRegistration, "-m", atlasImageMasked, "-r", brainImageRoughMasked, "-o",
               brainImagePrefix + "_aff.nrrd", "-O", brainImagePrefix + "_aff_tr.txt", "-i",
               brainImagePrefix + "_rig_tr.txt", "--sp", "3", "--ot", "2"] + pyramidOptions
    call(command)

    command = [animaDenseSVFBMRegistration, "-r", brainImageRoughMasked, "-m", brainImagePrefix + "_aff.nrrd", "-o",
               brainImagePrefix + "_nl.nrrd", "-O", brainImagePrefix + "_nl_tr.nrrd", "--sr", "1"] + pyramidOptions
    call(command)

    command = [animaApplyTransformSerie, "-i", iccImage, "-t", brainImagePrefix + "_nl_tr.xml", "-g", brainImage, "-o",
               brainImagePrefix + "_brainMask.nrrd", "-n", "nearest"]
    call(command)

    command = [animaMaskImage, "-i", brainImage, "-m", brainImagePrefix + "_brainMask.nrrd", "-o",
               brainImagePrefix + "_masked.nrrd"]
    call(command)
else:
    copyfile(brainImageRoughMasked,brainImagePrefix + "_masked.nrrd")
    copyfile(brainImagePrefix + "_rough_brainMask.nrrd",brainImagePrefix + "_brainMask.nrrd")

map(os.remove, glob.glob("*_rig*"))
map(os.remove, glob.glob("*_aff*"))
map(os.remove, glob.glob("*_nl*"))
map(os.remove, glob.glob("*_rough*"))

