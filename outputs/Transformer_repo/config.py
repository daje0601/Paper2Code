"""config.py

This module defines the central configuration for the project.
It reads from a YAML file "config.yaml" (if available) and exposes configuration
parameters in a structured and type‐safe manner. All other modules (dataset_loader.py,
model.py, trainer.py, evaluation.py, utils.py, main.py) import this file to ensure
consistency and reproducibility.

The configuration includes training parameters, model hyperparameters,
inference settings for machine translation, parsing experiment settings, dataset details,
and hardware specifications.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any

# ---------------------------
# Training Configuration
# ---------------------------
@dataclass(frozen=True)
class AdamConfig:
    beta1: float = 0.9
    beta2: float = 0.98
    epsilon: float = 1e-9

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'AdamConfig':
        return AdamConfig(
            beta1=float(data.get('beta1', 0.9)),
            beta2=float(data.get('beta2', 0.98)),
            epsilon=float(data.get('epsilon', 1e-9))
        )

@dataclass(frozen=True)
class TrainingConfig:
    total_steps: int = 100000
    warmup_steps: int = 4000
    optimizer: str = "adam"
    adam: AdamConfig = field(default_factory=AdamConfig)
    learning_rate_schedule: str = "d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)"
    dropout_rate: float = 0.1
    batch_tokens: int = 25000

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'TrainingConfig':
        adam_data = data.get('adam', {})
        return TrainingConfig(
            total_steps=int(data.get('total_steps', 100000)),
            warmup_steps=int(data.get('warmup_steps', 4000)),
            optimizer=str(data.get('optimizer', "adam")),
            adam=AdamConfig.from_dict(adam_data),
            learning_rate_schedule=str(data.get('learning_rate_schedule',
                                                  "d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)")),
            dropout_rate=float(data.get('dropout_rate', 0.1)),
            batch_tokens=int(data.get('batch_tokens', 25000))
        )

# ---------------------------
# Model Configuration
# ---------------------------
@dataclass(frozen=True)
class ModelConfig:
    type: str = "transformer_base"
    d_model: int = 512
    num_layers: int = 6
    d_ff: int = 2048
    num_heads: int = 8
    d_k: int = 64
    d_v: int = 64
    positional_encoding: str = "sinusoidal"
    share_embedding: bool = True

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ModelConfig':
        return ModelConfig(
            type=str(data.get('type', "transformer_base")),
            d_model=int(data.get('d_model', 512)),
            num_layers=int(data.get('num_layers', 6)),
            d_ff=int(data.get('d_ff', 2048)),
            num_heads=int(data.get('num_heads', 8)),
            d_k=int(data.get('d_k', 64)),
            d_v=int(data.get('d_v', 64)),
            positional_encoding=str(data.get('positional_encoding', "sinusoidal")),
            share_embedding=bool(data.get('share_embedding', True))
        )

# ---------------------------
# Inference (Translation) Configuration
# ---------------------------
@dataclass(frozen=True)
class InferenceConfig:
    beam_size: int = 4
    length_penalty: float = 0.6
    max_output_length_offset: int = 50

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'InferenceConfig':
        return InferenceConfig(
            beam_size=int(data.get('beam_size', 4)),
            length_penalty=float(data.get('length_penalty', 0.6)),
            max_output_length_offset=int(data.get('max_output_length_offset', 50))
        )

# ---------------------------
# Parsing Experiment Configuration
# ---------------------------
@dataclass(frozen=True)
class ParsingModelConfig:
    type: str = "transformer_parsing"
    num_layers: int = 4
    d_model: int = 1024

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ParsingModelConfig':
        return ParsingModelConfig(
            type=str(data.get('type', "transformer_parsing")),
            num_layers=int(data.get('num_layers', 4)),
            d_model=int(data.get('d_model', 1024))
        )

@dataclass(frozen=True)
class ParsingVocabConfig:
    wsj: int = 16000
    semi_supervised: int = 32000

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ParsingVocabConfig':
        return ParsingVocabConfig(
            wsj=int(data.get('wsj', 16000)),
            semi_supervised=int(data.get('semi_supervised', 32000))
        )

@dataclass(frozen=True)
class ParsingInferenceConfig:
    beam_size: int = 21
    length_penalty: float = 0.3
    max_output_length_offset: int = 300

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ParsingInferenceConfig':
        return ParsingInferenceConfig(
            beam_size=int(data.get('beam_size', 21)),
            length_penalty=float(data.get('length_penalty', 0.3)),
            max_output_length_offset=int(data.get('max_output_length_offset', 300))
        )

@dataclass(frozen=True)
class ParsingConfig:
    model: ParsingModelConfig = field(default_factory=ParsingModelConfig)
    vocab_size: ParsingVocabConfig = field(default_factory=ParsingVocabConfig)
    inference: ParsingInferenceConfig = field(default_factory=ParsingInferenceConfig)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ParsingConfig':
        model_data = data.get('model', {})
        vocab_data = data.get('vocab_size', {})
        inference_data = data.get('inference', {})
        return ParsingConfig(
            model=ParsingModelConfig.from_dict(model_data),
            vocab_size=ParsingVocabConfig.from_dict(vocab_data),
            inference=ParsingInferenceConfig.from_dict(inference_data)
        )

# ---------------------------
# Data Configuration
# ---------------------------
@dataclass(frozen=True)
class WMTConfig:
    dataset_size: str = "4.5M sentence pairs"
    tokenizer: str = "BPE"
    vocab_size: int = 37000

    @staticmethod
    def from_dict(data: Dict[str, Any], default_tokenizer: str, default_vocab: int) -> 'WMTConfig':
        return WMTConfig(
            dataset_size=str(data.get('dataset_size', "4.5M sentence pairs")),
            tokenizer=str(data.get('tokenizer', default_tokenizer)),
            vocab_size=int(data.get('vocab_size', default_vocab))
        )

@dataclass(frozen=True)
class WSJConfig:
    dataset: str = "Penn Treebank WSJ"
    training_size: str = "40K sentences"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'WSJConfig':
        return WSJConfig(
            dataset=str(data.get('dataset', "Penn Treebank WSJ")),
            training_size=str(data.get('training_size', "40K sentences"))
        )

@dataclass(frozen=True)
class SemiSupervisedParsingConfig:
    dataset: str = "BerkleyParser and high-confidence corpus"
    training_size: str = "17M sentences"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'SemiSupervisedParsingConfig':
        return SemiSupervisedParsingConfig(
            dataset=str(data.get('dataset', "BerkleyParser and high-confidence corpus")),
            training_size=str(data.get('training_size', "17M sentences"))
        )

@dataclass(frozen=True)
class DataConfig:
    wmt_2014_en_de: WMTConfig = field(default_factory=lambda: WMTConfig())
    wmt_2014_en_fr: WMTConfig = field(default_factory=lambda: WMTConfig(tokenizer="wordpiece", vocab_size=32000))
    wsj: WSJConfig = field(default_factory=WSJConfig)
    semi_supervised_parsing: SemiSupervisedParsingConfig = field(default_factory=SemiSupervisedParsingConfig)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'DataConfig':
        wmt_de_data = data.get('wmt_2014_en_de', {})
        wmt_fr_data = data.get('wmt_2014_en_fr', {})
        wsj_data = data.get('wsj', {})
        semi_data = data.get('semi_supervised_parsing', {})
        wmt_2014_en_de = WMTConfig.from_dict(wmt_de_data, default_tokenizer="BPE", default_vocab=37000)
        wmt_2014_en_fr = WMTConfig.from_dict(wmt_fr_data, default_tokenizer="wordpiece", default_vocab=32000)
        wsj_config = WSJConfig.from_dict(wsj_data)
        semi_config = SemiSupervisedParsingConfig.from_dict(semi_data)
        return DataConfig(
            wmt_2014_en_de=wmt_2014_en_de,
            wmt_2014_en_fr=wmt_2014_en_fr,
            wsj=wsj_config,
            semi_supervised_parsing=semi_config
        )

# ---------------------------
# Hardware Configuration
# ---------------------------
@dataclass(frozen=True)
class HardwareConfig:
    gpus: int = 8
    gpu_type: str = "NVIDIA P100"
    base_model_training_time: str = "12 hours"
    big_model_training_time: str = "3.5 days"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'HardwareConfig':
        return HardwareConfig(
            gpus=int(data.get('gpus', 8)),
            gpu_type=str(data.get('gpu_type', "NVIDIA P100")),
            base_model_training_time=str(data.get('base_model_training_time', "12 hours")),
            big_model_training_time=str(data.get('big_model_training_time', "3.5 days"))
        )

# ---------------------------
# Global Configuration
# ---------------------------
@dataclass(frozen=True)
class Config:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    parsing: ParsingConfig = field(default_factory=ParsingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Config':
        training_data = data.get('training', {})
        model_data = data.get('model', {})
        inference_data = data.get('inference', {})
        parsing_data = data.get('parsing', {})
        data_section = data.get('data', {})
        hardware_data = data.get('hardware', {})

        training_cfg = TrainingConfig.from_dict(training_data)
        model_cfg = ModelConfig.from_dict(model_data)
        inference_cfg = InferenceConfig.from_dict(inference_data)
        parsing_cfg = ParsingConfig.from_dict(parsing_data)
        data_cfg = DataConfig.from_dict(data_section)
        hardware_cfg = HardwareConfig.from_dict(hardware_data)

        config_instance = Config(
            training=training_cfg,
            model=model_cfg,
            inference=inference_cfg,
            parsing=parsing_cfg,
            data=data_cfg,
            hardware=hardware_cfg
        )

        # Validate that model.d_model equals num_heads * d_k.
        if config_instance.model.d_model != config_instance.model.num_heads * config_instance.model.d_k:
            raise ValueError(
                f"Invalid model configuration: d_model ({config_instance.model.d_model}) != "
                f"num_heads ({config_instance.model.num_heads}) * d_k ({config_instance.model.d_k})."
            )
        return config_instance

def load_config(config_path: str = "config.yaml") -> Config:
    """
    Loads configuration parameters from a YAML file and returns a Config instance.
    If the file does not exist or is empty, default values are used.
    """
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            yaml_data = yaml.safe_load(f)
            if yaml_data is None:
                yaml_data = {}
    else:
        yaml_data = {}
    return Config.from_dict(yaml_data)

# Global configuration instance used by other modules.
CONFIG: Config = load_config()

if __name__ == "__main__":
    # For debugging: print the loaded configuration.
    print("Loaded Configuration:")
    print(CONFIG)
