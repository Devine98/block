import collections

from block.global_conf import ModelConfig

PROJECT = "spelling_correction"
PRETRAINED_MODELS = collections.OrderedDict()
PRETRAINED_MODELS["mlp_corrector_s"] = ModelConfig(
    name="mlp_corrector_s",
    model_type="pkl",
    project=PROJECT,
)
PRETRAINED_MODELS["mlp_corrector"] = ModelConfig(
    name="mlp_corrector",
    model_type="pkl",
    project=PROJECT,
)
PRETRAINED_MODELS["mlp_corrector_test"] = ModelConfig(
    name="mlp_corrector_test",
    model_type="pkl",
    project=PROJECT,
)
