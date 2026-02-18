#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q shared
#SBATCH -J pretrain
#SBATCH --mail-user=jderm@nersc.gov
#SBATCH --mail-type=ALL
#SBATCH -A m4506
#SBATCH -t 01:30:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

module load conda

conda activate wind

export ACCELERATE=" \
  --config_file /global/homes/j/jderm/.cache/huggingface/accelerate/default_config.yaml \
"

export ARGS=" \
	--validate \
	--experiment-string=depth4_heads4_lr3e-4-val \
        --checkpoint-path=/pscratch/sd/j/jderm/runs/depth4_heads4_lr3e-4_epoch3/ \
	--device=accelerate \
	--hrrr-path=/pscratch/sd/j/jderm/HRRR_reduced/ \
	--era5-path=/pscratch/sd/j/jderm/ERA5_reduced/ \
	--train-file=./era_hrrr_perlmutter_train.txt \
	--val-file=./era_hrrr_perlmutter_val.txt \
	--batch-size=1 \
	--grad-accumulation-steps=1 \
	--epochs=600 \
	--encoder-learning-rate=3e-5 \
	--decoder-learning-rate=3e-4 \
	--encoder-weight-decay=1e-8 \
	--decoder-weight_decay=1e-8 \
	--encoder-depth=8 \
	--encoder-dim=768 \
	--encoder-channels=12 \
	--encoder-heads=4 \
	--encoder-mlp_dim=2304 \
	--decoder-depth=4 \
	--decoder-dim=768 \
	--decoder-channels=2 \
	--decoder-heads=4 \
	--decoder-mlp_dim=2304 \
"

srun  -n 1 -c 32 --cpu_bind=cores -G 1 --gres=gpu:1 accelerate launch $ACCELERATE ./train_acc.py $ARGS 2>&1 | tee inference_val2.out

