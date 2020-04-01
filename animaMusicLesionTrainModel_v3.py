import sys
import os
import argparse
from CNN_training_tools.build_model import cascade_model
from CNN_training_tools.base import train_cascaded_model

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

parser = argparse.ArgumentParser(
    prog='animaMusicLesionTrainModel_v3.py',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Train the cascaded CNN to segment MS Lesions.')

parser.add_argument('-i', '--inputDirectory', required=True, help='Training data: the directory containing all training subjects.')
parser.add_argument('-m', '--model', default="t1_flair_1608_ce_noNorm_upsampleAnima_rev1", help='Model name.')
parser.add_argument('-f', '--flair', default="FLAIR_masked-upsampleAnima.nii.gz", help='FLAIR file name.')
parser.add_argument('-t', '--t1', default="T1_masked-upsampleAnima.nii.gz", help='T1 file name.')
parser.add_argument('-T', '--t2', default="T2_masked-upsampleAnima.nii.gz", help='T2 file name.')
parser.add_argument('-l', '--label', default="Consensus-upsampleAnima.nii.gz", help='Label file name.')

args = parser.parse_args()

train_subjects = args.inputDirectory

modelName = args.model
t1Image = args.t1
t2Image = args.t2
flairImage = args.flair
cImage = args.label

def music_lesion_train_model(animaExtraDataDir, train_subjects, t1Image, t2Image, flairImage, cImage, modelName):
    
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

print(modelName)
music_lesion_train_model(animaExtraDataDir, train_subjects, t1Image, t2Image, flairImage, cImage, modelName)
