import os
from subprocess import call, check_output

# Data additional preprocessing, uses as inputs the prprocessed images from animaMSExamRegistration.py
def music_lesion_additional_preprocessing(animaDir,animaExtraDataDir,tmpFolder,t1Image,flairImage,cImage,maskImage,nbThreads):

    # Anima commands
    animaPyramidalBMRegistration = os.path.join(animaDir,"animaPyramidalBMRegistration")
    animaApplyTransformSerie = os.path.join(animaDir,"animaApplyTransformSerie")
    animaMaskImage = os.path.join(animaDir,"animaMaskImage")
    animaTransformSerieXmlGenerator = os.path.join(animaDir,"animaTransformSerieXmlGenerator")
    animaDenseSVFBMRegistration = os.path.join(animaDir,"animaDenseSVFBMRegistration")
    animaNyulStandardization = os.path.join(animaDir,"animaNyulStandardization")
    animaMorphologicalOperations = os.path.join(animaDir,"animaMorphologicalOperations")
    animaConvertImage = os.path.join(animaDir,"animaConvertImage")
#    animaKMeansStandardization = os.path.join(animaDir,"animaKMeansStandardization")
    animaImageResolutionChanger = os.path.join(animaDir,"animaImageResolutionChanger") # not 1mm isotropic
    
    # Atlases
    olivierAtlasDir = os.path.join(animaExtraDataDir,"olivier-atlas")
    olivierAtlasImageMasked = os.path.join(olivierAtlasDir,"ATLAS-masked.nrrd")
    olivierAtlasMask = os.path.join(olivierAtlasDir,"ATLAS-mask.nrrd")

    uspioAtlasDir = os.path.join(animaExtraDataDir,"uspio-atlas")
    uspioAtlasDir_T1 = os.path.join(uspioAtlasDir,"scalar-space","T1")
    uspioAtlasDir_FLAIR = os.path.join(uspioAtlasDir,"scalar-space","FLAIR")
    uspioAtlasDir_RefSpace = os.path.join(uspioAtlasDir,"scalar-space","space-ref")

    # Decide on whether to use large image setting or small image setting
    command = [animaConvertImage, "-i", t1Image, "-I"]
    convert_output = check_output(command, universal_newlines=True)
    size_info = convert_output.split('\n')[1].split('[')[1].split(']')[0]
    large_image = False
    for i in range(0, 3):
        size_tmp = int(size_info.split(', ')[i])
        if size_tmp > 350:
            large_image = True
            break

    pyramidOptions = ["-p", "4", "-l", "1"]
    if large_image:
        pyramidOptions = ["-p", "5", "-l", "2"]

    # register T1 atlas to T1 image
    command = [animaPyramidalBMRegistration,"-m",olivierAtlasImageMasked,"-r",t1Image,"-o",os.path.join(tmpFolder,"atlas_aff.nrrd"),"-O",os.path.join(tmpFolder,"atlas_aff_tr.txt"),
               "-I","2","--sp","3","--ot","2","-T",nbThreads] + pyramidOptions
    call(command)
    command = [animaDenseSVFBMRegistration,"-r",t1Image,"-m",os.path.join(tmpFolder,"atlas_aff.nrrd"),"-o",os.path.join(tmpFolder,"atlas_nl.nrrd"),"-O",os.path.join(tmpFolder,"atlas_nl_tr.nrrd"),"--sr","1","-T",nbThreads] + pyramidOptions
    call(command)
    command = [animaTransformSerieXmlGenerator,"-i",os.path.join(tmpFolder,"atlas_aff_tr.txt"),"-i",os.path.join(tmpFolder,"atlas_nl_tr.nrrd"),"-o",os.path.join(tmpFolder,"atlas_nl_tr.xml")]
    call(command)

    # apply transform to atlas mask
    command = [animaApplyTransformSerie,"-i",olivierAtlasMask,"-t", os.path.join(tmpFolder,"atlas_nl_tr.xml"),"-g", t1Image,"-o", os.path.join(tmpFolder,"mask.nrrd"),"-n","nearest","-p",nbThreads]
    call(command)

    # intersect mask
    command = [animaMaskImage, "-i", maskImage, "-m",os.path.join(tmpFolder,"mask.nrrd"), "-o", os.path.join(tmpFolder,"mask.nrrd")]
    call(command)

    # register wm, gm, csf maps
    command = [animaApplyTransformSerie,"-i", os.path.join(olivierAtlasDir, "ATLAS-wm_masked.nrrd"),"-t", os.path.join(tmpFolder,"atlas_nl_tr.xml"),"-g", t1Image, "-o", os.path.join(tmpFolder,"ATLAS-wm_masked-reg.nrrd")]
    call(command)
    command = [animaApplyTransformSerie,"-i", os.path.join(olivierAtlasDir, "ATLAS-gm_masked.nrrd"),"-t", os.path.join(tmpFolder,"atlas_nl_tr.xml"),"-g", t1Image, "-o", os.path.join(tmpFolder,"ATLAS-gm_masked-reg.nrrd")]
    call(command)
    command = [animaApplyTransformSerie,"-i", os.path.join(olivierAtlasDir, "ATLAS-csf_masked.nrrd"),"-t", os.path.join(tmpFolder,"atlas_nl_tr.xml"),"-g", t1Image, "-o", os.path.join(tmpFolder, "ATLAS-csf_masked-reg.nrrd")]
    call(command)
    
    # register T1 control to T1 image - to normalize within the same mask 
    command = [animaPyramidalBMRegistration,"-m",os.path.join(uspioAtlasDir_T1, "T1_1.nrrd"),"-r",t1Image,"-o",os.path.join(tmpFolder,"atlas_aff.nrrd"),"-O",os.path.join(tmpFolder,"atlas_aff_tr.txt"),
               "-I","2","--sp","3","--ot","2","-T",nbThreads] + pyramidOptions
    call(command)
    command = [animaDenseSVFBMRegistration,"-r",t1Image,"-m",os.path.join(tmpFolder,"atlas_aff.nrrd"),"-o",os.path.join(tmpFolder,"atlas_nl.nrrd"),"-O",os.path.join(tmpFolder,"atlas_nl_tr.nrrd"),"--sr","1","-T",nbThreads] + pyramidOptions
    call(command)
    command = [animaTransformSerieXmlGenerator,"-i",os.path.join(tmpFolder,"atlas_aff_tr.txt"),"-i",os.path.join(tmpFolder,"atlas_nl_tr.nrrd"),"-o",os.path.join(tmpFolder,"atlas_nl_tr.xml")]
    call(command)
    
    # apply transform to controls
    command = [animaApplyTransformSerie,"-i",os.path.join(uspioAtlasDir_RefSpace,"brain-mask_intersected.nrrd"),"-t", os.path.join(tmpFolder,"atlas_nl_tr.xml"),"-g", t1Image,"-o", os.path.join(tmpFolder,"brain-mask_intersected_reg.nrrd"),"-n","nearest","-p",nbThreads]
    call(command)
    command = [animaApplyTransformSerie,"-i",os.path.join(uspioAtlasDir_T1, "T1_1.nrrd"),"-t", os.path.join(tmpFolder,"atlas_nl_tr.xml"),"-g", t1Image,"-o", os.path.join(tmpFolder,"T1_1_reg.nrrd"),"-p",nbThreads]
    call(command)
    command = [animaApplyTransformSerie,"-i",os.path.join(uspioAtlasDir_FLAIR, "FLAIR_1.nrrd"),"-t", os.path.join(tmpFolder,"atlas_nl_tr.xml"),"-g", t1Image,"-o", os.path.join(tmpFolder,"FLAIR_1_reg.nrrd"),"-p",nbThreads]
    call(command)
    
    # intersect control mask and patient mask  
    command = [animaMaskImage, "-i", maskImage, "-m",os.path.join(tmpFolder,"brain-mask_intersected_reg.nrrd"), "-o", os.path.join(tmpFolder,"brain-mask_intersected_reg_inters.nrrd")]
    call(command)
    # erode brain mask (to avoid border lesions)
    command = [animaMorphologicalOperations, "-i", os.path.join(tmpFolder, "brain-mask_intersected_reg_inters.nrrd"), "-o", os.path.join(tmpFolder, "mask-er.nrrd"), "-a", "er", "-r", "1.5"]
    call(command)
    
    # apply mask 
    command=[animaMaskImage, "-i", os.path.join(tmpFolder,"FLAIR_1_reg.nrrd"), "-m", os.path.join(tmpFolder,"mask-er.nrrd"), "-o", os.path.join(tmpFolder, "FLAIR_1_reg_masked.nrrd")]
    call(command)
    command=[animaMaskImage, "-i", os.path.join(tmpFolder,"T1_1_reg.nrrd"), "-m", os.path.join(tmpFolder,"mask-er.nrrd"), "-o", os.path.join(tmpFolder, "T1_1_reg_masked.nrrd")]
    call(command)
    command=[animaMaskImage, "-i", flairImage, "-m", os.path.join(tmpFolder,"mask-er.nrrd"), "-o", os.path.join(tmpFolder, "FLAIR_masked.nrrd")]
    call(command)
    command=[animaMaskImage, "-i", t1Image, "-m", os.path.join(tmpFolder,"mask-er.nrrd"), "-o", os.path.join(tmpFolder, "T1_masked.nrrd")]
    call(command)
    
#    # normalize k-means
#    command=[animaKMeansStandardization, "-r", os.path.join(tmpFolder, "FLAIR_1_reg_masked.nrrd"), "-m", os.path.join(tmpFolder, "FLAIR_masked.nrrd"),
#             "-R", os.path.join(tmpFolder,"mask-er.nrrd"), "-M", os.path.join(tmpFolder,"mask-er.nrrd"),
#             "-o", os.path.join(tmpFolder, "FLAIR-normed.nrrd")]
#    call(command)
#    command=[animaKMeansStandardization, "-r", os.path.join(tmpFolder, "T1_1_reg_masked.nrrd"), "-m", os.path.join(tmpFolder, "T1_masked.nrrd"),
#             "-R", os.path.join(tmpFolder,"mask-er.nrrd"), "-M", os.path.join(tmpFolder,"mask-er.nrrd"),
#             "-o", os.path.join(tmpFolder, "T1-normed.nrrd")]
#    call(command)
    
    # normalize nyul
    command=[animaNyulStandardization, "-r", os.path.join(tmpFolder, "FLAIR_1_reg_masked.nrrd"), "-m", os.path.join(tmpFolder, "FLAIR_masked.nrrd"),
             "-o", os.path.join(tmpFolder, "FLAIR-normed_nyul.nrrd")]
    call(command)
    command=[animaNyulStandardization, "-r", os.path.join(tmpFolder, "T1_1_reg_masked.nrrd"), "-m", os.path.join(tmpFolder, "T1_masked.nrrd"),
             "-o", os.path.join(tmpFolder, "T1-normed_nyul.nrrd")]
    call(command)
         
    # Resample images to get 1x1x1 mm3 images
    command = [animaImageResolutionChanger, "-i", os.path.join(tmpFolder,"FLAIR_masked.nrrd"), "-o", os.path.join(tmpFolder,"FLAIR_masked-upsampleAnima.nii.gz"),
               "-x","1","-y","1","-z","1"]
    call(command)
    command = [animaTransformSerieXmlGenerator,"-i",os.path.join(animaExtraDataDir,"id.txt"),"-o",os.path.join(tmpFolder,"id.xml")]
    call(command)
    command = [animaApplyTransformSerie, "-i", os.path.join(tmpFolder,"T1_masked.nrrd"), "-g", os.path.join(tmpFolder,"FLAIR_masked-upsampleAnima.nii.gz"),
               "-t",os.path.join(tmpFolder,"id.xml"),"-o",os.path.join(tmpFolder,"T1_masked-upsampleAnima.nii.gz")]
    call(command)
    
    # Lesion
    command = [animaApplyTransformSerie, "-i", cImage, "-g", os.path.join(tmpFolder,"FLAIR_masked-upsampleAnima.nii.gz"),
                   "-t",os.path.join(tmpFolder,"id.xml"),"-o",os.path.join(tmpFolder,"Consensus-upsampleAnima.nii.gz"),"-n", "nearest"]
    call(command)

    # Resample images to get 1x1x1 mm3 images
    command = [animaImageResolutionChanger, "-i", os.path.join(tmpFolder,"FLAIR-normed_nyul.nrrd"), "-o", os.path.join(tmpFolder,"FLAIR-normed_nyul-upsampleAnima.nii.gz"),
               "-x","1","-y","1","-z","1"]
    call(command)
    command = [animaTransformSerieXmlGenerator,"-i",os.path.join(animaExtraDataDir,"id.txt"),"-o",os.path.join(tmpFolder,"idn.xml")]
    call(command)
    command = [animaApplyTransformSerie, "-i", os.path.join(tmpFolder,"T1-normed_nyul.nrrd"), "-g", os.path.join(tmpFolder,"FLAIR-normed_nyul-upsampleAnima.nii.gz"),
               "-t",os.path.join(tmpFolder,"idn.xml"),"-o",os.path.join(tmpFolder,"T1-normed_nyul-upsampleAnima.nii.gz")]
    call(command)
