

export ACCELERATE=" \
  --cpu \
  --num_machines 1 \
  --num_processes 4 \
  --num_cpu_threads_per_process 2 \
  --debug  \
"


export ARGS=" \
	--experiment-string=val-test \
	--pretrain \
	--device=accelerate \
	--hrrr-path=/pscratch/sd/j/jderm/HRRR_reduced/ \
	--era5-path=/pscratch/sd/j/jderm/ERA5_reduced/ \
	--train-file=./era_hrrr_alter_perlmutter.txt \
	--val-file=./era_hrrr_alter_perlmutter.txt \
	--batch-size=24 \
	--epochs=500 \
	--encoder-learning-rate=1e-3 \
	--decoder-learning-rate=1e-3 \
	--encoder-weight-decay=1e-4 \
	--decoder-weight_decay=1e-4 \
	--encoder-depth=8 \
	--encoder-dim=768 \
	--encoder-channels=16 \
	--encoder-heads=4 \
	--encoder-mlp_dim=2304 \
	--decoder-depth=8 \
	--decoder-dim=768 \
	--decoder-channels=2 \
	--decoder-heads=4 \
	--decoder-mlp_dim=2304 \
"

accelerate launch $ACCELERATE train_acc.py $ARGS 2>&1 | tee val-test.out

