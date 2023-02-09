import collections

from block.global_conf import ModelConfig

PROJECT = "translation"
PRETRAINED_MODELS = collections.OrderedDict()
PRETRAINED_MODELS["opus-mt-mul-en"] = ModelConfig(
    name="opus-mt-mul-en",
    model_type="gz.tar",
    project=PROJECT,
)
