import collections

from block.global_conf import ModelConfig

PROJECT = "bricks"
PRETRAINED_MODELS = collections.OrderedDict()
PRETRAINED_MODELS["bert_51m"] = ModelConfig(
    name="bert_51m",
    model_type="gz.tar",
    project=PROJECT,
)
PRETRAINED_MODELS["bert_16m"] = ModelConfig(
    name="bert_16m",
    model_type="gz.tar",
    project=PROJECT,
)
PRETRAINED_MODELS["bert_124m"] = ModelConfig(
    name="bert_124m",
    model_type="gz.tar",
    project=PROJECT,
)

PRETRAINED_TOKENIZERS = collections.OrderedDict()
PRETRAINED_TOKENIZERS["bert_tokenizer"] = ModelConfig(
    name="bert_tokenizer",
    model_type="pkl",
    project=PROJECT,
)
