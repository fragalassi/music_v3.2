import os
from keras.models import load_model
from CNN_training_tools.base import test_scan
from CNN_training_tools.metrics import generalised_dice_loss, jaccard_distance_loss

def music_lesion_core_processing(animaExtraDataDir,t1Image,t2Image,flairImage,modelName,tmpFolder):
                     
    custom_objects = {'generalised_dice_loss': generalised_dice_loss, 'jaccard_distance_loss': jaccard_distance_loss}

    model_1 = dict()
    model_2 = dict()

    model_1['net'] = load_model(os.path.join("ms_lesion_models", modelName + "_1.h5"), custom_objects=custom_objects)
    model_2['net'] = load_model(os.path.join("ms_lesion_models", modelName + "_2.h5"), custom_objects=custom_objects)
    model_1['net'].load_weights(os.path.join("ms_lesion_models", modelName + "_weights_1.h5"), by_name=True)
    model_2['net'].load_weights(os.path.join("ms_lesion_models", modelName + "_weights_2.h5"), by_name=True)
    
    model = [model_1, model_2]

    options = dict()

    options['test_folder'] = ''
    options['test_scan'] = ''
    options['experiment'] = ''
    options['patch_size'] = (11, 11, 11)
    options['min_th'] = 0.5
    options['randomize_train'] = True
    options['fully_conv'] = True
    options['debug'] = False
    options['batch_size'] = 128
    options['net_verbose'] = 2
    options['min_error'] = 0.5
    options['l_min'] = 5
    options['t_bin'] = 0.5

    test_data = dict()
    test_data['Patient'] = dict()
    test_data['Patient']['T1'] = t1Image
    test_data['Patient']['T2'] = t2Image
    test_data['Patient']['FLAIR'] = flairImage

    # First layer of CNN
    options['test_name'] = os.path.join(tmpFolder, modelName, modelName + "_prob_0.nii.gz")
    first_candidates = test_scan(model[0], test_data, options, save_nifti=True)

    options['test_name'] = os.path.join(tmpFolder, modelName, modelName + "_prob_1.nii.gz")
    test_scan(model[1], test_data, options, save_nifti=True, candidate_mask=(first_candidates > 0.8))
                        
## test script
#test_folder='/temp_dd/igrida-fs1/fgalassi/testing/'
#test_folder='/temp_dd/igrida-fs1/fgalassi/export-music/'
#animaExtraDataDir='/temp_dd/igrida-fs1/fgalassi/MUSIC_rev2/'
#
#modelName = 't1_flair_1608_ce_noNorm_rev1'
#
#list_of_scans = os.listdir(test_folder)
#list_of_scans.sort()
#
#for scan in list_of_scans:
#     
#     print(scan)
#     subj_folder = os.path.join(test_folder, scan)
#     exp_folder = os.path.join(subj_folder, modelName)
#     
#     if not os.path.exists(exp_folder):
#         os.mkdir(exp_folder)
#         
#     cnnImage = os.path.join(exp_folder, modelName)
#
#     t1Image = os.path.join(subj_folder, "T1_masked-upsampleAnima.nii.gz")
#     flairImage = os.path.join(subj_folder, "FLAIR_masked-upsampleAnima.nii.gz")
#         
#     music_lesion_core_processing(animaExtraDataDir,cnnImage,t1Image,flairImage,modelName)
     