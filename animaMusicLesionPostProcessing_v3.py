import os
from subprocess import call

def music_lesion_post_processing(animaDir, animaExtraDataDir, tmpFolder, outputImage, cnnImage,
                                 flairImage, atlasWMImage, atlasGMImage, atlasCSFImage, maskImage, nbThreads):
                                     
    # anima tools - some not used
    animaThrImage = os.path.join(animaDir, "animaThrImage")
    animaMaskImage = os.path.join(animaDir, "animaMaskImage")
    animaConnectedComponents = os.path.join(animaDir, "animaConnectedComponents")
    animaInfluenceZones = os.path.join(animaDir, "animaInfluenceZones")
    animaRemoveTouchingBorder = os.path.join(animaDir, "animaRemoveTouchingBorder")
    animaFillHoleImage = os.path.join(animaDir, "animaFillHoleImage")
    animaApplyTransformSerie = os.path.join(animaDir, "animaApplyTransformSerie")
    
    #Resample back to initial resolution
#    command = [animaTransformSerieXmlGenerator,"-i",os.path.join(animaExtraDataDir,"id.txt"),"-o",os.path.join(tmpFolder,"id.xml")]
#    call(command)
    command = [animaApplyTransformSerie, "-i", cnnImage, "-o", outputImage, "-t", os.path.join(tmpFolder, "id.xml"),
               "-g", flairImage, "-n", "linear"]
    call(command)

    # Threshold hard at 0.5 to get detections
    command = [animaThrImage, "-i", outputImage, "-t", "0.5", "-o", outputImage]
    call(command)

    # thresh wm gm csf
    command = [animaThrImage, "-i", atlasWMImage, "-t", "0.3", "-o",
               os.path.join(tmpFolder, 'ATLAS-wm_mask-reg-thr.nrrd')]
    call(command)
    command = [animaThrImage, "-i", atlasGMImage, "-t", "0.8", "-o",
               os.path.join(tmpFolder, 'ATLAS-gm_mask-reg-inv.nrrd'), "-I"]
    call(command)
    command = [animaThrImage, "-i", atlasCSFImage, "-t", "0.2", "-o",
               os.path.join(tmpFolder, 'ATLAS-csf_mask-reg-inv.nrrd'), "-I"]
    call(command)

    # remove les in csf and gm & keep wm lesions
    command = [animaMaskImage, "-i", os.path.join(tmpFolder, "ATLAS-csf_mask-reg-inv.nrrd"), "-m",
               outputImage, "-o", outputImage]
    call(command)
    command = [animaMaskImage, "-i", os.path.join(tmpFolder, "ATLAS-gm_mask-reg-inv.nrrd"), "-m",
               outputImage, "-o", outputImage]
    call(command)
    command = [animaMaskImage, "-i", os.path.join(tmpFolder, "ATLAS-wm_mask-reg-thr.nrrd"), "-m",
               outputImage, "-o", outputImage]
    call(command)

    # remove les touching outer mask border
    command = [animaInfluenceZones, "-i", outputImage, "-o", os.path.join(tmpFolder, 'segm-label.nrrd')]
    call(command)
    command = [animaRemoveTouchingBorder, "-i", os.path.join(tmpFolder, 'segm-label.nrrd'), "-m",
               maskImage, "-o", os.path.join(tmpFolder, 'segm-label.nrrd'), "-L", "-T", nbThreads]
    call(command)

    # fill lesions
    command = [animaFillHoleImage, "-i", outputImage, "-o", outputImage]
    call(command)

    # remove small lesions, final output
    command = [animaConnectedComponents, "-i", outputImage, "-m", "6", "-o", outputImage]
    call(command)
    command = [animaThrImage, "-i", outputImage, "-t", "0", "-o", outputImage]
    call(command)
    

