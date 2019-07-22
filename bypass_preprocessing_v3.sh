#!/bin/bash
SCRIPTDIR=/temp_dd/igrida-fs1/fgalassi/MUSIC_rev2/MUSIC_v3
echo "$SCRIPTDIR"

DATADIR=/temp_dd/igrida-fs1/fgalassi/training
#DATADIR=/temp_dd/igrida-fs1/fgalassi/UNC_train_Part1
#DATADIR=/temp_dd/igrida-fs1/fgalassi/export-music
#DATADIR=/temp_dd/igrida-fs1/fgalassi/testing
#DATADIR=/temp_dd/igrida-fs1/fgalassi/brainM24/set1
#DATADIR=/temp_dd/igrida-fs1/fgalassi/brainM24/set2_includingSomeM0

# Load modules in cluster
. /etc/profile.d/modules.sh
set -xv

module load cuDNN/v7.0.4
module load cuda/9.0.176

# Activate the py virtual environnement
. /udd/fgalassi/myVE/bin/activate

chmod +x $SCRIPTDIR/*.py

for patient in $DATADIR/*; do	

    if [ -d "$patient" ]; then
	 patientID=$(basename "$patient")
	 echo "$patientID"	
 	
: '
	#PREPROCESSING	
	#training/testing MICCAI16
	FLAIR=$DATADIR/"$patientID"/3DFLAIR.nii.gz
	T1=$DATADIR/"$patientID"/3DT1.nii.gz
	T1gd=$DATADIR/"$patientID"/3DT1GADO.nii.gz

	#export-music 2019
	FLAIR=$DATADIR/"$patientID"/flair.nrrd
	T1=$DATADIR/"$patientID"/t1.nrrd

	#training MICCAI 2008
	FLAIR=$DATADIR/"$patientID"/FLAIR.nii.gz
	T1=$DATADIR/"$patientID"/T1.nii.gz

	#M024
	FLAIR=$DATADIR/"$patientID"/br-flair.nii.gz
	T1=$DATADIR/"$patientID"/br-t1.nii.gz

	python3 $SCRIPTDIR/animaMSExamPreparation.py -r $FLAIR -f $FLAIR -t $T1  -o $DATADIR/"$patientID"/ 
'

	#ADDITIONAL PREPROCESSING
	#training/testing MICCAI16
	FLAIR=$DATADIR/"$patientID"/3DFLAIR_preprocessed.nrrd
	T1=$DATADIR/"$patientID"/3DT1_preprocessed.nrrd
	MASK=$DATADIR/"$patientID"/3DFLAIR_brainMask.nrrd
	C=$DATADIR/"$patientID"/Consensus.nii.gz

: '
	#export-music 2019
	FLAIR=$DATADIR/"$patientID"/flair_preprocessed.nrrd
	T1=$DATADIR/"$patientID"/t1_preprocessed.nrrd
	MASK=$DATADIR/"$patientID"/flair_brainMask.nrrd

	#training MICCAI 2008
	FLAIR=$DATADIR/"$patientID"/FLAIR_preprocessed.nrrd
	T1=$DATADIR/"$patientID"/T1_preprocessed.nrrd
	MASK=$DATADIR/"$patientID"/FLAIR_brainMask.nrrd
	C=$DATADIR/"$patientID"/UNC_train_lesion_byCHB.nii.gz

	#M024
	FLAIR=$DATADIR/"$patientID"/br-flair_preprocessed.nrrd
	T1=$DATADIR/"$patientID"/br-t1_preprocessed.nrrd
	MASK=$DATADIR/"$patientID"/br-flair_brainMask.nrrd
'
	PYTHONHASHSEED=0 python3 $SCRIPTDIR/animaMusicLesionSegmentation_v3.py -f $FLAIR -t $T1 -m $MASK -n 6 -o $DATADIR/"$patientID"/

    fi
done


