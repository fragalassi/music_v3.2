#!/bin/bash

# Use absolute path for igrida
SCRIPTDIR=./
echo "$SCRIPTDIR"

# Load modules in cluster
. /etc/profile.d/modules.sh
set -xv

module load cuDNN/v7.0.4
module load cuda/9.0.176

# Activate the py virtual environnement
. ./bin/activate

chmod +x $SCRIPTDIR/*.py

PYTHONHASHSEED=0 python3 $SCRIPTDIR/main.py -d emisepBrainSegmentationData.json -n 6
