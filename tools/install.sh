#!/bin/bash

# make sure only first task per node installs stuff, others wait
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
if [[ $SLURM_LOCALID == 0 ]]; then
    # setup environment
    if [ ! -d "venv" ]; then
        python3 -m venv --system-site-packages venv
        source venv/bin/activate
        pip install -e .
    fi

    # activate environment
    source venv/bin/activate

    # export huggingface cache path
    export HF_HOME=./_cache/huggingface

    # tell other tasks we are done installing
    touch "${DONEFILE}"
else
    # wait until packages are installed
    while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi

# this runs your wrapped command
"$@"