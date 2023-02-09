import os
from typing import List, Union

from transformers import pipeline

from block.translation.modeling_utils import PreTrainedModel


class Translator(PreTrainedModel):
    """Translator .

    using Helsinki-NLP.
    """

    def __init__(self) -> None:
        super().__init__()

        self.model = None

    def _load(self, model: str) -> None:
        """Load  state dict from local model path or dict.

        Args:
            model (str):
                Model file need to be loaded.
                A string, the path of a pretrained model.

        Raises:
            ValueError: str model should be a path!
        """

        if isinstance(model, str):
            if os.path.isdir(model):
                self._load_from_dir(model)
            elif os.path.isfile(model):
                dir = os.path.join(self._tmpdir.name, "translation")
                if os.path.exists(dir):
                    pass
                else:
                    os.mkdir(dir)
                self._unzip2dir(model, dir)
                self._load_from_dir(dir)
            else:
                raise ValueError("""str model should be a path!""")

        else:
            raise ValueError("""str model should be a path!""")

    def _load_from_dir(self, model_dir: str) -> None:
        """Set model params from `model_file`.

        Args:
            model_dir (str):
                Dir containing model params.
        """

        self.model = pipeline(
            task="translation_mul_to_en", model=model_dir, tokenizer=model_dir
        )

    def translate(self, text: Union[str, List[str]]) -> List[str]:
        """translate `text`.

        Args:
            text (Union[str,List[str]]):
            The text need to be translated.

        Returns:
            List[str]:
        """
        if not text:
            text = ""
        if isinstance(text, str):
            text = [text]
        text = [t[:400] for t in text]

        return [t["translation_text"] for t in self.model(text)]

    def __call__(self, text: Union[str, List[str]]) -> List[str]:
        return self.translate(text)
