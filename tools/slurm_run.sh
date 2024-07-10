PARTITION=${PARTITION:-H100,A100-80GB,A100-40GB,RTXA6000}
GPUS=${GPUS:-1}
CPUS=${CPUS:-4}
MEM=${MEM:-48GB}

srun -K \
    -p $PARTITION  \
    --mem=$MEM \
    --ntasks=1 \
    --gpus-per-task=$GPUS \
    --cpus-per-task=$CPUS \
    --container-image=./env.sqsh \
    --container-mounts=/home/$USER:/home/$USER,/netscratch:/netscratch,"`pwd`":"`pwd`,/ds-av:/ds-av" \
    --container-workdir="`pwd`" \
    --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    --pty /bin/bash
