#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH -q premium
#SBATCH -J pretrain
#SBATCH --mail-user=jderm@nersc.gov
#SBATCH --mail-type=ALL
#SBATCH -A m4723
#SBATCH -t 10:00:00
#SBATCH -D /global/homes/j/jderm/


#OpenMP settings:
export OMP_NUM_THREADS=128
export OMP_PLACES=threads
export OMP_PROC_BIND=false

module load conda

conda activate wind

# training setup
GPUS_PER_NODE=4

# so that processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29508
NNODES=$SLURM_NNODES
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export NCCL_IB_DISABLE=1

export HF_HOME=$SCRATCH/cache/huggingface
mkdir -p $HF_HOME

###################
### set network ###
###################

head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

export LAUNCHER="accelerate launch \
  --multi_gpu \
  --num_machines $NNODES \
  --num_processes $WORLD_SIZE \
  --main_process_ip "$MASTER_ADDR" \
  --main_process_port $MASTER_PORT \
  --machine_rank \$SLURM_PROCID \
  --rdzv_conf rdzv_backend=c10d \
  --max_restarts 0 \
  --tee 3 \
"

export BASEDIR="/people/derm950/lisa/s2s_backup/"

export DATAFILE=$BASEDIR"paper_model_scripts/s2s_v4/masking_tests/assets/buoy_train_5k.csv"
export VALFILE=$BASEDIR"goldstandard/gold_npy_buoy_val_sorted.csv"
export JOBDONE="COMPLETE"
export DOTOUT=".OUT"

echo $LAUNCHER

	export FINETUNE=" \
	--experiment-string=dec_supervised_5k_lisa \
	--device=accelerate \
	--npy \
	--hrrr-path=$BASEDIR/hrrr_data/ \
	--era5-path=$BASEDIR/era5_data/ \
	--checkpoint-path=$BASEDIR \
	--train-file=$DATAFILE \
	--val-file=$VALFILE \
	--batch-size=5 \
	--finetune1 \
	--grad-accumulation-steps=2 \
	--epochs=120 \
	--encoder-learning-rate=3e-5 \
	--decoder-learning-rate=3e-5 \
	--encoder-weight-decay=3e-3 \
	--decoder-weight_decay=3e-3 \
	--encoder-depth=8 \
	--encoder-dim=768 \
	--encoder-channels=15 \
	--encoder-heads=4 \
	--encoder-mlp_dim=2304 \
	--decoder-depth=4 \
	--decoder-dim=392 \
	--decoder-channels=2 \
	--decoder-heads=4 \
	--decoder-mlp_dim=1152 \
	--encoder-masking-ratio=0.20 \
	--decoder-masking-ratio=0.95 \
	--beta1=0.9 \
	--beta2=0.999 \
	--eps=1e-8 \
	--grad-norm=1 \
"

SRUN_ARGS=" \
	--wait=60 \
	--kill-on-bad-exit=1 \
	"

clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "echo $DATAFILE;  echo $JOBDONE ; $LAUNCHER $BASEDIR/train_acc.py $FINETUNE 2>&1 | tee /global/homes/j/jderm/dec_sup_5k_lisa\$SLURM_PROCID$DOTOUT"

echo "END TIME: $(date)"

