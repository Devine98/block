import collections
import os
import pickle
import tarfile
import tempfile
from typing import Dict, Optional, Union

import boto3
import botocore
import hdfs

from block.global_conf import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    BUCKET_NAME,
    HDFS_ADDRESS,
    MINIO_URL,
    TEMP_PATH,
)


class PreTrainedModule:

    """Pretrained module for all modules.

    Basic class takes care of storing the configuration of the models
    and handles methods for loading ,downloading and saving.

    Attributes:
        pretrained_models  (Optional[collections.OrderedDict], optional):
            Pretrained model list.
            Defaults to None.
    """

    _TEMP_PATH = TEMP_PATH

    def __init__(
        self,
        pretrained_models: Optional[collections.OrderedDict] = None,
    ) -> None:
        # Model temp dir for documents or state dicts
        if not os.path.exists(TEMP_PATH):
            os.mkdir(TEMP_PATH)
        self._tmpdir = tempfile.TemporaryDirectory(prefix=f"{TEMP_PATH}/")
        if pretrained_models is None:
            pretrained_models = collections.OrderedDict()
        self._pretrained_models = pretrained_models

    def from_pretrained(
        self,
        model_name: Optional[Union[str, Dict]] = None,
        download_path: Optional[str] = None,
    ) -> None:
        """Load state dict of `model_name` from hdfs or local path

        Args:
            model_name (Optional[Union[str, Dict]], optional):
                Predtrained model need to be loaded.
                Can be either:
                    - A string, the `model_name` of a pretrained model.
                    - A path to a `directory` containing model weights.
                    - A state dict containing model weights.
                    - None . which means first model in `_pretrained_models`
                Defaults to None.
            download_path (Optional[str], optional):
                Path the model should be downloaded to.
                If None pretrained model will downloaded to `download_path`.
                Else pretrained model will downloaded to temp path.
                Defaults to None.
        """
        if download_path is None:
            download_path = self._tmpdir.name

        if model_name is None:
            pretrained_models = list(self._pretrained_models.keys())
            if not pretrained_models:
                raise IndexError(
                    """No pretrained model found,
                                 please input a valid `model_name`!"""
                )
            model_name = pretrained_models[0]

        if isinstance(model_name, str):
            if model_name in self._pretrained_models:
                model_config = self._pretrained_models[model_name]
                model = f"{download_path}/{model_config.name}.{model_config.model_type}"
                if os.path.exists(model):
                    pass
                else:
                    try:
                        self._s3_download(key=model_config.s3_url, file_name=model)
                        print(f"{model_config.name}  successfully downloaded from s3")
                    except botocore.exceptions.ClientError:
                        self._hdfs_download(
                            local_path=download_path,
                            hdfs_path=model_config.hdfs_url,
                        )
                        print(f"{model_config.name}  successfully downloaded from hdfs")
            else:
                model = model_name

        else:
            model = model_name
        self._load(model)

    def _load(self, model: Union[str, Dict]) -> None:
        """Load  state dict from local model path or dict.

        Args:
            model (Union[str, Dict]):
                Model file need to be loaded.
                Can be either:
                    - A string, the path of a pretrained model.
                    - A state dict containing model weights.
        """

        pass

    def _load_pkl(self, path: str) -> Dict:
        with open(path, "rb") as f:
            file = pickle.load(f)
        return file

    def _save_pkl(self, file: Dict, path: str) -> None:
        if os.path.exists(path):
            os.remove(path)
        with open(path, "wb") as f:
            pickle.dump(file, f)

    def _zip_dir(self, dir: str, path: str) -> None:
        tar = tarfile.open(path, "w")
        for files in os.listdir(dir):
            tar.add(os.path.join(dir, files), arcname=files)
        tar.close()

    def _unzip2dir(self, file: str, dir: Optional[str] = None) -> None:
        if dir is None:
            dir = self._tmpdir.name
        if not os.path.isdir(dir):
            raise ValueError("""`dir` shoud be a dir!""")
        tar = tarfile.open(file, "r")
        tar.extractall(path=dir)
        tar.close()

    def _hdfs_download(
        self, local_path: str, hdfs_path: str, hdfs_address: str = HDFS_ADDRESS
    ) -> None:
        """Download data from hdfs

        Args:
            local_path (str):
                Path for saving local data
            hdfs_path (str):
                Path of hdfs data.
            hdfs_address (str, optional):
                Defaults to HDFS_ADDRESS.
        """
        client = hdfs.InsecureClient(hdfs_address)
        client.download(
            hdfs_path=hdfs_path,
            local_path=local_path,
            overwrite=True,
        )

    def _s3_download(
        self,
        key: str,
        file_name: str,
        endpoint_url: str = MINIO_URL,
        aws_access_key_id: str = AWS_ACCESS_KEY_ID,
        aws_secret_access_key: str = AWS_SECRET_ACCESS_KEY,
    ) -> None:
        minio = boto3.resource(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        obj = minio.Object(BUCKET_NAME, key)
        obj.download_file(file_name)
