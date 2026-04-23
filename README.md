# LISA: Language Inspired Super Resolution Architecture

This repository contains the code associated with the manuscript:

> **A Language-Inspired Super-Resolution Architecture (LISA) For U.S. West Coast Low-Level Winds**
> Jack Dermigny, Sha Feng, Ye Liu, Larry K. Berg at PNNL
> Submitted to *Artificial Intelligent for the Earth System (AIES)*
<!-- > *[Journal Name]*, [Year]. DOI: [https://doi.org/XXXX] -->

LISA is a deep learning framework for statistical downscaling of near-surface wind speed from ERA5 reanalysis (~31 km) to HRRR resolution (~3 km). It uses a Vision Transformer (ViT) encoder pre-trained with Masked Autoencoding (MAE) followed by supervised fine-tuning, with an optional U-Net baseline for comparison.

---

## Repository Structure

```
lisa/
├── train_acc.py          # Main training script (MAE pre-training + ViT fine-tuning, multi-GPU via Accelerate)
├── train_unet.py         # U-Net baseline training script
├── models/
│   ├── models.py         # ERA5 encoder/decoder and upscaler architectures
│   ├── mae.py            # Masked Autoencoder (MAE) module
│   ├── vit.py            # Vision Transformer (ViT) backbone
│   └── unet.py           # U-Net architecture
├── utils/
│   ├── arguments.py      # Training, model, and optimizer parameter dataclasses
│   └── forecast_metrics.py  # Evaluation metrics and visualization utilities
├── normalize.py          # Input normalization utilities
├── normalize_ft.py       # Fine-tuning normalization utilities
├── normalize2.py         # Additional normalization utilities
├── inference.sh          # SLURM script for inference/validation on NERSC Perlmutter
├── launch.sh             # Launch script for local/CPU multi-process runs
├── test_run.sh           # Script for test runs
├── wind.yml              # Conda environment specification
└── requirements.txt      # Minimal pip dependencies
```

---

## Environment Setup

```bash
conda env create -f wind.yml
conda activate wind
```

This environment requires a CUDA-capable GPU. The code was developed and tested with:
- Python 3.11
- PyTorch 2.2.1
- CUDA 11.8

---

## Data

The model trains on paired ERA5 and HRRR data:

| Dataset | Resolution | Format | Notes |
|---------|-----------|--------|-------|
| [ERA5](https://cds.climate.copernicus.eu/) | ~31 km | NetCDF / `.npy` | Near-surface wind fields |
| [HRRR](https://rapidrefresh.noaa.gov/hrrr/) | ~3 km | NetCDF / `.npy` | High-Resolution Rapid Refresh |

Expected directory layout:
```
/data/
├── ERA5/          # ERA5 input files
├── HRRR/          # HRRR target files
├── train_files.txt  # List of training sample paths (one per line)
└── val_files.txt    # List of validation sample paths
```

---

## Usage

### 1. Pre-training (MAE)

```bash
accelerate launch train_acc.py \
    --pretrain \
    --era5-path /path/to/ERA5/ \
    --hrrr-path /path/to/HRRR/ \
    --train-file ./train_files.txt \
    --val-file ./val_files.txt \
    --epochs 500 \
    --batch-size 24 \
    --encoder-depth 8 \
    --encoder-dim 768 \
    --encoder-heads 4 \
    --experiment-string my_pretrain_run
```

### 2. Fine-tuning (Stage 1)

```bash
accelerate launch train_acc.py \
    --finetune1 \
    --checkpoint-path /path/to/pretrained_checkpoint/ \
    --era5-path /path/to/ERA5/ \
    --hrrr-path /path/to/HRRR/ \
    --train-file ./train_files.txt \
    --val-file ./val_files.txt \
    --experiment-string my_finetune_run
```

### 3. Fine-tuning (Stage 2)

Replace `--finetune1` with `--finetune2` and point `--checkpoint-path` to the stage 1 checkpoint.

### 4. Validation / Inference

```bash
accelerate launch train_acc.py \
    --validate \
    --checkpoint-path /path/to/finetuned_checkpoint/ \
    --era5-path /path/to/ERA5/ \
    --hrrr-path /path/to/HRRR/ \
    --train-file ./train_files.txt \
    --val-file ./val_files.txt \
    --batch-size 1
```

See `inference.sh` for a full SLURM example configured for NERSC Perlmutter.

### Key Arguments

| Argument | Description |
|----------|-------------|
| `--era5-path` | Path to ERA5 input data directory |
| `--hrrr-path` | Path to HRRR target data directory |
| `--train-file` | Text file listing training samples |
| `--val-file` | Text file listing validation samples |
| `--checkpoint-path` | Path to load/save model checkpoints |
| `--encoder-depth` | Number of ViT encoder layers |
| `--encoder-dim` | ViT embedding dimension |
| `--encoder-heads` | Number of attention heads |
| `--batch-size` | Training batch size |
| `--epochs` | Number of training epochs |
| `--pretrain` | Run MAE pre-training |
| `--finetune1` | Run fine-tuning stage 1 |
| `--finetune2` | Run fine-tuning stage 2 |
| `--validate` | Run validation/inference only |

---

## Pretrained Weights

[To be released upon acceptance — link to Zenodo/HuggingFace will be added here.]

---

## License

Copyright Battelle Memorial Institute 2026. Released under the [BSD-2-Clause License](LICENSE).

---

## Citation

If you use this code, please cite:

```bibtex
@article{[citekey],
  title   = {A Language-Inspired Super-Resolution Architecture (LISA) For U.S. West Coast Low-Level Winds},
  author  = {Jack Dermigny, Sha Feng, Ye Liu, Larry K. Berg },
  journal = {Artificial Intelligent for the Earth System (AIES) },
  year    = {[Year]},
  doi     = {[DOI]}
}
```

---

## Acknowledgments

This material is based upon work supported by the U.S. Department of Energy, Office of Science, Energy Earthshot Initiative as part of the "Addressing Challenges in Energy: Floating Wind in a Changing Climate (ACE-FWICC)” at Pacific Northwest National Laboratory (PNNL) under contract #KJ0406010/KP1601013/81823. The initial effort was inspired by an AI Hackathon organized by the Atmospheric, Climate, and Earth Sciences (ACES) Division at PNNL (Chen 2025). 
This research used resources of the National Energy Research Scientific Computing Center (NERSC), a Department of Energy Office of Science User Facility using NERSC award BER-ERCAP0026987 and This research used resources of the National Energy Research Scientific Computing Center (NERSC), a Department of Energy User Facility using NERSC award DDR-ERCAP0030520 through AI4Sci@NERSC. A portion of the research was performed using resources available through Research Computing at PNNL. PNNL is operated by Battelle for the U.S. Department of Energy under Contract DE-AC05-76RL01830. 


