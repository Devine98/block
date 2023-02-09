import collections

from block.global_conf import ModelConfig

PROJECT = "phrase_mining"
PRETRAINED_MODELS = collections.OrderedDict()
PRETRAINED_MODELS["freq_phrase_miner"] = ModelConfig(
    name="freq_phrase_miner",
    model_type="pkl",
    project=PROJECT,
)
