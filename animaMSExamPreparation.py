import argparse
import sys
import tempfile
import os
import shutil
from subprocess import call, check_output

if sys.version_info[0] > 2:
    import configparser as ConfParser
else:
    import ConfigParser as ConfParser

print('Parse ~/.anima/config.txt')

configFilePath = os.path.expanduser("~") + "/.anima/config.txt"
if not os.path.exists(configFilePath):
    print('Please create a configuration file for Anima python scripts. Refer to the README')
    quit()

configParser = ConfParser.RawConfigParser()
configParser.read(configFilePath)

animaDir = configParser.get("anima-scripts", 'anima')
# animaExtraDataDir = configParser.get("anima-scripts", 'extra-data-root')
# animaScriptsDir="/udd/fgalassi/Anima-Scripts"

# Anima commands
animaPyramidalBMRegistration = os.path.join(animaDir, "animaPyramidalBMRegistration")
animaMaskImage = os.path.join(animaDir, "animaMaskImage")
animaNLMeans = os.path.join(animaDir, "animaNLMeans")
animaN4BiasCorrection = os.path.join(animaDir, "animaN4BiasCorrection")
animaConvertImage = os.path.join(animaDir, "animaConvertImage")
#animaBrainExtractionScript = os.path.join(animaScriptsDir, "brain_extraction", "animaAtlasBasedBrainExtraction.py")

def extractExtension(fileName):
    fileNamePrefix = os.path.splitext(fileName)[0]
    if os.path.splitext(fileName)[1] == '.gz':
        fileNamePrefix = os.path.splitext(fileNamePrefix)[0]
    return fileNamePrefix

def process(reference, flair, t1, t1_gd="", t2="", outputFolder=tempfile.gettempdir()):
    
    refImage = reference

    images = [flair, t1]
    if t1_gd != "":
        images.append(t1_gd)
    if t2 is not None and t2 != "":
        images.append(t2)

    tmpFolder = outputFolder

    brainExtractionCommand = ["python3", os.path.join(os.path.dirname(__file__), "animaAtlasBasedBrainExtraction.py"), "-i", refImage, "-S"]
    call(brainExtractionCommand)

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

    refImagePrefix = extractExtension(refImage)

    brainMask = refImagePrefix + "_brainMask.nrrd"

    # Main loop
    for image in images:
        inputPrefix = extractExtension(image)

        registeredDataFile = os.path.join(tmpFolder, "SecondImage_registered.nrrd")
        rigidRegistrationCommand = [animaPyramidalBMRegistration, "-r", refImage, "-m", image, "-o",
                                    registeredDataFile] + pyramidOptions
        call(rigidRegistrationCommand)

        unbiasedSecondImage = os.path.join(tmpFolder, "SecondImage_unbiased.nrrd")
        biasCorrectionCommand = [animaN4BiasCorrection, "-i", registeredDataFile, "-o", unbiasedSecondImage, "-B", "0.3"]
        call(biasCorrectionCommand)

        # nlmSecondImage = os.path.join(tmpFolder, "SecondImage_unbiased_nlm.nrrd")
        # nlmCommand = [animaNLMeans, "-i", unbiasedSecondImage, "-o", nlmSecondImage, "-n", "3"]
        # call(nlmCommand)

        outputPreprocessedFile = os.path.join(tmpFolder, inputPrefix + "_preprocessed.nrrd")
        # secondMaskCommand = [animaMaskImage, "-i", nlmSecondImage, "-m", brainMask, "-o", outputPreprocessedFile]
        secondMaskCommand = [animaMaskImage, "-i", unbiasedSecondImage, "-m", brainMask, "-o", outputPreprocessedFile]
        call(secondMaskCommand)

    for image in images:
        inputPrefix = extractExtension(image)
        
        tempFileNames = ['_aff.nrrd', '_aff_tr.txt', '_brainMask.nrrd', '_masked.nrrd', '_nl.nrrd', '_nl_tr.nrrd', '_nl_tr.xml', '_preprocessed.nrrd', '_rig.nrrd', '_rig_tr.txt', '_rough_brainMask.nrrd', '_rough_masked.nrrd']
        for tempFileName in tempFileNames:
            if os.path.isfile(inputPrefix + tempFileName):
                shutil.move(inputPrefix + tempFileName, os.path.join(tmpFolder, os.path.basename(inputPrefix + tempFileName)))
    
    if os.path.isfile(brainMask) and not os.path.exists(os.path.join(tmpFolder, brainMask)):
        shutil.move(brainMask, tmpFolder)

    #shutil.rmtree(tmpFolder)
