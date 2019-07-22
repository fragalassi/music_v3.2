#!/bin/bash
SCRIPTDIR=/temp_dd/igrida-fs1/fgalassi/MUSIC_rev2/MUSIC_v3/
echo "$SCRIPTDIR"

DATADIR=/temp_dd/igrida-fs1/fgalassi/training
#DATADIR=/temp_dd/igrida-fs1/fgalassi/UNC_train_Part1

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
	#training MICCAI16
	FLAIR=$DATADIR/"$patientID"/Raw/3DFLAIR.nii.gz
	T1=$DATADIR/"$patientID"/Raw/3DT1.nii.gz
	T1gd=$DATADIR/"$patientID"/Raw/3DT1GADO.nii.gz
	T2=$DATADIR/"$patientID"/Raw/T2.nii.gz

	#training MICCAI 2008
	FLAIR=$DATADIR/"$patientID"/FLAIR.nii.gz
	T1=$DATADIR/"$patientID"/T1.nii.gz


	python3 $SCRIPTDIR/animaMSExamPreparation.py -r $FLAIR -f $FLAIR -t $T1  -o $DATADIR/"$patientID"/ 
'

	#ADDITIONAL PREPROCESSING
	if [[ -f "$DATADIR/"$patientID"/ManualSegmentation/Consensus.nii.gz" ]]; then
		#training MICCAI16
		FLAIR=$DATADIR/"$patientID"/Raw/3DFLAIR_preprocessed.nrrd
		T1=$DATADIR/"$patientID"/Raw/3DT1_preprocessed.nrrd
		MASK=$DATADIR/"$patientID"/Raw/3DFLAIR_brainMask.nrrd
		C=$DATADIR/"$patientID"/ManualSegmentation/Consensus.nii.gz

	else 

		#training MICCAI 2008
		FLAIR=$DATADIR/"$patientID"/FLAIR_preprocessed.nrrd
		T1=$DATADIR/"$patientID"/T1_preprocessed.nrrd
		MASK=$DATADIR/"$patientID"/FLAIR_brainMask.nrrd
		C=$DATADIR/"$patientID"/UNC_train_lesion_byCHB.nii.gz
	fi

	echo "$C"

	#PTHONHASHSEED=0 python3 $SCRIPTDIR/animaMusicLesionSegmentationTraining_v3.py -f $FLAIR -t $T1 -m $MASK -c $C -n 6 -o $DATADIR/"$patientID"/

    fi
done

echo "training..."
PTHONHASHSEED=0 python3 $SCRIPTDIR/animaMusicLesionTrainModel_v3.py 






