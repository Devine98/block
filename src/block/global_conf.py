import os
from typing import Optional

TEMP_PATH = "/tmp/.sprite"
HDFS_ADDRESS = "http://dts-sg-hadoop01:50070/"
HDFS_PATH = "/workspace/ml/nlp/sprite"

MINIO_URL = os.getenv("MINIO_URL", default="http://api.minio.test.data.garenanow.com")
AWS_SECRET_ACCESS_KEY = os.getenv(
    "AWS_SECRET_ACCESS_KEY", default="vYXsIEVO64MbUBX6rbzHhKwvAbOg0Mbj"
)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", default="jobg3Zd1y3Gv1DJF")
BUCKET_NAME = os.getenv("BUCKET_NAME", default="ml-test")


class ModelConfig:
    """Basic class takes care of storing the configuration of the models

    Attributes:
        name (str):
            pretrained model name
        model_type (str):
            pretrained model type
            Defaults to "pkl"
        project (str):
            pretrained model project
        hdfs_url (Optional[str], optional):
            hdfs url for pretrained model
            Defaults to None.
        s3_url (Optional[str], optional):
            s2 url for pretrained model
            Defaults to None.


    """

    def __init__(
        self,
        name: str,
        model_type: str = "pkl",
        project: Optional[str] = None,
        hdfs_url: Optional[str] = None,
        s3_url: Optional[str] = None,
    ) -> None:
        self.name = name
        self.model_type = model_type
        self.project = project

        if hdfs_url is None:
            hdfs_url = f"{HDFS_PATH}/{project}/{name}.{model_type}"
        self.hdfs_url = hdfs_url

        if s3_url is None:
            s3_url = f"sprite/{project}/{name}.{model_type}"
        self.s3_url = s3_url
