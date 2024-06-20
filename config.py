import os
from typing import Optional

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "runs")


import os
from typing import Optional


class BaseConfig:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        temp_out_dir: str,
        model_dir: str = DEFAULT_MODEL_DIR,
    ) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.temp_out_dir = temp_out_dir
        self.model_dir = model_dir

    @property
    def input_dir(self) -> str:
        return self._input_dir

    @input_dir.setter
    def input_dir(self, value: str) -> None:
        self._input_dir = value
        self.input_mask_dir = os.path.join(self._input_dir, "masks/")

    @property
    def output_dir(self) -> str:
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value: str) -> None:
        self._output_dir = value
        self.seg_out_dir = os.path.join(self._output_dir, "seg_out/")
        self.det_out_dir = os.path.join(self._output_dir, "det_out/")
        self.output_tils_dir = os.path.join(self.output_dir, "tils_out/")

    @property
    def temp_out_dir(self) -> str:
        return self._temp_out_dir

    @temp_out_dir.setter
    def temp_out_dir(self, value: str) -> None:
        self._temp_out_dir = value

    @property
    def model_dir(self) -> Optional[str]:
        return self._model_dir

    @model_dir.setter
    def model_dir(self, value: Optional[str]) -> None:
        self._model_dir = value
        self.cell_model_dir = os.path.join(self._model_dir, "cell/weights_v2")
        self.tissue_model_dir = os.path.join(
            self._model_dir, "tissue/weights_v2"
        )

    def create_output_dirs(self) -> None:
        for directory in [
            self.temp_out_dir,
            self.seg_out_dir,
            self.det_out_dir,
            self.output_tils_dir,
        ]:
            if not os.path.exists(directory):
                print(f"creating dir: {directory}")
                os.makedirs(directory)


class Config(BaseConfig):
    def __init__(
        self,
        input_dir: str = "/media/u1910100/Extreme SSD/data/tiger/wsitils/images",
        output_dir: str = "/home/u1910100/Documents/Tiger_Data/prediction",
        model_dir: Optional[str] = DEFAULT_MODEL_DIR,
    ) -> None:
        super().__init__(
            input_dir,
            output_dir,
            os.path.join(output_dir, "temp_out/"),
            model_dir,
        )
        self.input_mask_dir = self.temp_out_dir


class Challenge_Config(BaseConfig):
    def __init__(
        self,
        input_dir: str = "/home/u1910100/Documents/Tiger_Data/testinput",
        output_dir: str = "/home/u1910100/Documents/Tiger_Data/output",
        temp_out_dir: str = "/home/u1910100/Documents/Tiger_Data/tempoutput",
        model_dir: Optional[str] = DEFAULT_MODEL_DIR,
    ) -> None:
        super().__init__(input_dir, output_dir, temp_out_dir, model_dir)
        self.seg_out_dir = os.path.join(
            output_dir, "images/breast-cancer-segmentation-for-tils/"
        )
        self.det_out_dir = output_dir
        self.output_tils_dir = output_dir
