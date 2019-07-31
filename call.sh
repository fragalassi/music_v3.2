#!/bin/bash
DATADIR=../testing

for patient in $DATADIR/*; do	

    if [ -d "$patient" ]; then
	 patientID=$(basename "$patient")
	 echo "$patientID"	
 	
	#MICCAI16
	FLAIR=$DATADIR/"$patientID"/3DFLAIR.nii.gz
	T1=$DATADIR/"$patientID"/3DT1.nii.gz

	python3 animaMSExamPreparation.py -r $FLAIR -f $FLAIR -t $T1  -o $DATADIR/"$patientID"/ 

	#ADDITIONAL PREPROCESSING
	#training/testing MICCAI16
	FLAIR=$DATADIR/"$patientID"/3DFLAIR_preprocessed.nrrd
	T1=$DATADIR/"$patientID"/3DT1_preprocessed.nrrd
	MASK=$DATADIR/"$patientID"/3DFLAIR_brainMask.nrrd
	C=$DATADIR/"$patientID"/Consensus.nii.gz

	python3 animaMusicLesionSegmentation_v3.py -f $FLAIR -t $T1 -m $MASK -n 6 -o $DATADIR/"$patientID"/

    fi
done


