from dataclasses import dataclass, fields

@dataclass
class TrainingParam:
    checkpoint_path:   str = ""
    experiment_string: str = "test"
    batch_size: int = 40
    grad_accumulation_steps: int = 1
    epochs:     int = 100
    seed:     int = 42
    pretrain:  bool = False
    finetune1: bool = False
    finetune2: bool = False
    validate:  bool = False
    npy:       bool = False
    device:     str = "cpu"
    era5_path:  str = ""
    hrrr_path:  str = ""
    train_file: str = ""
    val_file:   str = ""
    dataparallel: bool = False

    def __repr__(self):

        return_str = "\n"

        for field in fields(self):
            return_str += f"{field.name}\t{getattr(self, field.name)}\n"

        return return_str

@dataclass
class OptParam:
    encoder_learning_rate: float = 1e-3
    encoder_weight_decay:  float = 1e-2
    decoder_learning_rate: float = 1e-3
    decoder_weight_decay:  float = 1e-2
    grad_norm:  float = 10
    beta1:  float = 0.9
    beta2:  float = 0.999
    eps:  float = 1e-8

    def __repr__(self):

        return_str = "\n"

        for field in fields(self):
            return_str += f"{field.name}\t{getattr(self, field.name)}\n"

        return return_str

@dataclass
class ModelParam:
    encoder_depth:    int = 8
    encoder_dim:      int = 768
    encoder_channels: int = 12
    encoder_heads:    int = 4
    encoder_mlp_dim:  int = 3*768
    encoder_num_registers:  int = 0
    encoder_masking_ratio:  float = 0.60

    decoder_depth:    int = 8
    decoder_dim:      int = 768
    decoder_channels: int = 2
    decoder_heads:    int = 4
    decoder_mlp_dim:  int = 3*768
    decoder_masking_ratio:  float = 0.60

    def __repr__(self):

        return_str = "\n"

        for field in fields(self):
            return_str += f"{field.name}\t{getattr(self, field.name)}\n"

        return return_str

