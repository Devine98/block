import collections

from block.global_conf import ModelConfig

PROJECT = "toxic_detection"
PRETRAINED_MODELS = collections.OrderedDict()
PRETRAINED_MODELS["toxic_detector_bert_51m"] = ModelConfig(
    name="toxic_detector_bert_51m",
    model_type="gz.tar",
    project=PROJECT,
)
PRETRAINED_MODELS["toxic_detector_bert_16m"] = ModelConfig(
    name="toxic_detector_bert_16m",
    model_type="gz.tar",
    project=PROJECT,
)
PRETRAINED_MODELS["toxic_detector_bert_51m_by_region"] = ModelConfig(
    name="toxic_detector_bert_51m_by_region",
    model_type="gz.tar",
    project=PROJECT,
)
PRETRAINED_MODELS["toxic_detector_bert_16m_by_region"] = ModelConfig(
    name="toxic_detector_bert_16m_by_region",
    model_type="gz.tar",
    project=PROJECT,
)
