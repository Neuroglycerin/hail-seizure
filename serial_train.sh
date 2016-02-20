#!/usr/bin/env bash
#
# serial_train.sh
# Inputs
#   - Directory containing .json settings file
#   - Flag for generating predictions. Set to "-p" to do predict.py. Optional.

#==============================================================================
# Input handling
#==============================================================================

# strip trailing / slashes
DIR=${1%/};

#==============================================================================
# Main
#==============================================================================

NUM_FILES=$(ls -l $DIR/*.json | wc -l)
i=0

for file in $DIR/*.json
do
    ((i++))
	echo "== Processing ($i/$NUM_FILES): $file =="
    if [ "$2" == "-p" ]
    then
        python3 train.py -s $file &> /dev/null &&
            python3 predict.py -s $file &> /dev/null
    else
        python3 train.py -s $file &> /dev/null
    fi
    if (($?==0))
    then
    	echo "** Completed processing ($i/$NUM_FILES): $file **"
    else
    	echo "!! Failed processing $file !!"
    fi
done
