#!/bin/bash

SCRIPTDIR=/temp_dd/igrida-fs1/fgalassi/music_v3.2
echo "$SCRIPTDIR"

chmod +x $SCRIPTDIR/bypass_preprocessing_v3.sh

#oarsub -l {"host = 'igrida-abacus4.irisa.fr'"}/gpu_device=1,walltime=03:00:0 "$SCRIPTDIR/bypass_FG_testing.sh"

#oarsub -t besteffort -t idempotent -p "dedicated='none' or dedicated = 'serpico' or dedicated = 'linkmedia' or dedicated = 'sirocco' or dedicated = 'intuidoc'" -l {"gpu_model = 'Tesla P100'"}/gpu_device=1,walltime=03:00:0 "$SCRIPTDIR/bypass_preprocessing_v3.sh"

oarsub -t besteffort -t idempotent -p "dedicated='none' or dedicated = 'serpico' or dedicated = 'linkmedia' or dedicated = 'sirocco' or dedicated = 'intuidoc'" -l /gpu_device=1,walltime=03:00:0 "$SCRIPTDIR/bypass_preprocessing_v3.sh"

#oarsub -t besteffort -t idempotent -p "dedicated='none' or dedicated = 'serpico' or dedicated = 'linkmedia' or dedicated = 'sirocco' or dedicated = 'intuidoc'" -l {"host ='igrida-abacus6.irisa.fr' OR host ='igrida-abacus7.irisa.fr' OR host ='igrida-abacus8.irisa.fr' OR host ='igrida-abacus11.irisa.fr' OR host ='igrida-abacus10.irisa.fr' OR host ='igrida-abacus3.irisa.fr' OR host='igrida-abacus4.irisa.fr' OR host='igrida-abacus2.irisa.fr' OR host ='igrida-abacus9.irisa.fr'"}/gpu_device=1,walltime=01:0:0 "$SCRIPTDIR/bypass_preprocessing_v3.sh" 

