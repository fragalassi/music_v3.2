import os
import sys
import shutil
import tempfile

import configparser as ConfParser

import animaMusicLesionAdditionalPreprocessing_v3 as preproc
import animaMusicLesionCoreProcessing_v3 as coreproc
import animaMusicLesionPostProcessing_v3 as postproc

print('   parse ~/.anima/config.txt')

if sys.version_info[0] < 3:
    sys.exit('Python must be of version 3 or higher for this script.')


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

def process(flairImage, t1Image, t2Image, maskImage, cImage, outputFolder, nbThreads, training):

    tmpFolder=outputFolder

    #--------------------------------------- First perform additional preprocessing
    print('Starting additional preprocessing of data')
    preproc.music_lesion_additional_preprocessing(animaDir, animaExtraDataDir, tmpFolder, t1Image, t2Image, flairImage, maskImage, nbThreads, cImage, not training)

    if not training:
        #----------------------------------------- Then run core process over up images
        print('Done with additional preprocessing, starting core processing of data')
        modelName = "t1_flair_1608_ce_noNorm_upsampleAnima_rev1"
        if not(os.path.isdir(os.path.join(tmpFolder, modelName))):
            os.makedirs(os.path.join(tmpFolder, modelName))
        
        # Use unnormalized images
        # t1Image = os.path.join(tmpFolder, "T1_masked_upsampleAnima.nii.gz")
        # t2Image = os.path.join(tmpFolder, "T2_masked_upsampleAnima.nii.gz")
        # flairImage = os.path.join(tmpFolder, "FLAIR_masked_upsampleAnima.nii.gz")
        # Use normalized images
        t1Image = os.path.join(tmpFolder, "T1_masked_normed_nyul_upsampleAnima.nii.gz")
        t2Image = os.path.join(tmpFolder, "T2_masked_normed_nyul_upsampleAnima.nii.gz")
        flairImage = os.path.join(tmpFolder, "FLAIR_masked_normed_nyul_upsampleAnima.nii.gz")
        
        coreproc.music_lesion_core_processing(animaExtraDataDir, t1Image, t2Image, flairImage, modelName, tmpFolder)

        #------------------------------------------------------ Now run post-processing
        print('Done with core processing, starting post processing of data')

        maskImage = os.path.join(tmpFolder, "mask-er.nrrd")
        atlasWMImage = os.path.join(tmpFolder, "ATLAS-wm_masked-reg.nrrd")
        atlasGMImage = os.path.join(tmpFolder, "ATLAS-gm_masked-reg.nrrd")
        atlasCSFImage = os.path.join(tmpFolder, "ATLAS-csf_masked-reg.nrrd")
        flairImage = args.flair
        cnnImage = os.path.join(tmpFolder, modelName, modelName + "_prob_1.nii.gz")

        outputImage = os.path.join(tmpFolder, modelName, modelName + "_segm.nii.gz")
        postproc.music_lesion_post_processing(animaDir, animaExtraDataDir, tmpFolder, outputImage, cnnImage, flairImage,
                                            atlasWMImage, atlasGMImage, atlasCSFImage, maskImage, nbThreads)
                                            
        ##---------------------------------------------------- Evaluate segm performance
        #SEGPERF="/udd/fgalassi/dev/SegPerfAnalyzer_build/SegPerfAnalyzer-build/SegPerfAnalyzer"
        #command=[SEGPERF, "-r",  os.path.join(tmpFolder, "Consensus.nii.gz"), "-i", outputImage, "-o", os.path.join(tmpFolder, modelName, modelName + "_segm_perf"),"-s", "-l"]
        #call(command)
        #    
        #shutil.rmtree(tmpFolder)
