import sys
import os
import argparse
from CNN_training_tools.build_model import cascade_model
from CNN_training_tools.base import train_cascaded_model

# Read the anima extra-data directory path in ~/.anima/config.txt
if sys.version_info[0] > 2:
    import configparser as ConfParser
else:
    import ConfigParser as ConfParser

configFilePath = os.path.expanduser("~") + "/.anima/config.txt"
if not os.path.exists(configFilePath):
    print('Please create a configuration file for Anima python scripts (~/.anima/config.txt). Refer to the Anima Scripts README')
    quit()

configParser = ConfParser.RawConfigParser()
configParser.read(configFilePath)

animaExtraDataDir = configParser.get("anima-scripts", 'extra-data-root')

# Train the model

def music_lesion_train_model(train_subjects, t1Image="T1_masked_normed_nyul_upsampleAnima.nii.gz", t2Image="T2_masked_normed_nyul_upsampleAnima.nii.gz", flairImage="FLAIR_masked_normed_nyul_upsampleAnima.nii.gz", cImage="Consensus_upsampleAnima.nii.gz", modelName="t1_flair_1608_ce_noNorm_upsampleAnima_rev1"):
    
    options = {}
    
    options['randomize_train'] = True
    options['fully_convolutional'] = False
    options['debug'] = True
    options['full_train'] = True
    options['load_weights'] = False

#    options['loss'] = 'generalised_dice_loss' # longer training
#    options['loss'] = 'jaccard_distance_loss' # longer training
    options['loss'] = 'categorical_crossentropy'

    options['patience'] = 20
    options['max_epochs'] = 200
    options['train_split'] = 0.4     
    options['patch_size'] = (11, 11, 11)
    options['min_th'] = 0.5 # to select candidate voxels on flair
    options['batch_size'] = 128
    options['net_verbose'] = 2
    
    options['min_error'] = 0.5 # minm res the model can handle in ml
    options['l_min'] = 5
    options['t_bin'] = 0.5

    options['train_folder'] = train_subjects
    options['weight_paths'] = animaExtraDataDir
    options['experiment'] = modelName
    options['modalities'] = ['T1','T2','FLAIR']
    options['x_names'] = [t1Image, t2Image, flairImage]
    options['y_names'] = [cImage]

    list_of_scans = os.listdir(options['train_folder'])
    list_of_scans.sort()
    modalities = options['modalities']
    x_names = options['x_names']
    y_names = options['y_names']
    
    # training data
    train_x_data = {f: {m: os.path.join(options['train_folder'], f, n) for m, n in zip(modalities, x_names)}
                       for f in list_of_scans}
    train_y_data = {f: os.path.join(options['train_folder'], f, y_names[0]) for f in list_of_scans}
    
    # initialize model
    model = cascade_model(options)
    # train model
    model = train_cascaded_model(model, train_x_data, train_y_data, options)
    
    # saves the architecture
    model[0]['net'].save(os.path.join(animaExtraDataDir,"ms_lesion_models", modelName+'_1.h5'))
    model[1]['net'].save(os.path.join(animaExtraDataDir,"ms_lesion_models", modelName+'_2.h5'))
    
    # saves the weights
    model[0]['net'].save_weights(os.path.join(animaExtraDataDir,"ms_lesion_models",modelName+'_weights_1.h5'),overwrite=True)
    model[1]['net'].save_weights(os.path.join(animaExtraDataDir,"ms_lesion_models",modelName+'_weights_2.h5'),overwrite=True)
