import os

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "runs")


class Config:
    def __init__(
        self,
        input_dir="/media/u1910100/Extreme SSD/data/tiger/wsitils/images",
        output_dir="/home/u1910100/Documents/Tiger_Data/prediction",
        model_dir=DEFAULT_MODEL_DIR,
    ) -> None:
        self.output_dir = output_dir
        self.input_dir = input_dir

        self.temp_out_dir = os.path.join(self.output_dir, "temp_out/")
        self.seg_out_dir = os.path.join(self.output_dir, "seg_out/")
        self.det_out_dir = os.path.join(self.output_dir, "det_out/")
        self.output_tils_dir = os.path.join(self.output_dir, f"tils/")
        # self.create_dirs()

        self.model_dir = model_dir
        self.cell_model_dir = os.path.join(self.model_dir, "cell/weights_v2")
        self.tissue_model_dir = os.path.join(
            self.model_dir, "tissue/weights_v2"
        )

    def create_dirs(self):
        if not os.path.exists(self.temp_out_dir):
            os.makedirs(self.temp_out_dir)
        if not os.path.exists(self.seg_out_dir):
            os.makedirs(self.seg_out_dir)
        if not os.path.exists(self.det_out_dir):
            os.makedirs(self.det_out_dir)
        if not os.path.exists(self.output_tils_dir):
            os.makedirs(self.output_tils_dir)


class Challenge_Config:
    def __init__(
        self,
        input_dir="/home/u1910100/Documents/Tiger_Data/testinput",
        output_dir="/home/u1910100/Documents/Tiger_Data/output",
        temp_out_dir="/home/u1910100/Documents/Tiger_Data/tempoutput",
        model_dir=DEFAULT_MODEL_DIR,
    ) -> None:
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.temp_out_dir = temp_out_dir

        self.input_mask_dir = os.path.join(
            self.input_dir, "images/"
        )  # Input mask dir
        self.seg_out_dir = os.path.join(
            self.output_dir, "images/breast-cancer-segmentation-for-tils/"
        )
        self.det_out_dir = self.output_dir
        self.output_tils_dir = self.output_dir
        # self.create_dirs()

        self.model_dir = model_dir
        self.cell_model_dir = os.path.join(self.model_dir, "cell/weights_v2")
        self.tissue_model_dir = os.path.join(
            self.model_dir, "tissue/weights_v2"
        )

    def create_output_dirs(self):
        if not os.path.exists(self.temp_out_dir):
            print(f"creating dir: {self.temp_out_dir}")
            os.makedirs(self.temp_out_dir)

        if not os.path.exists(self.seg_out_dir):
            print(f"creating dir: {self.seg_out_dir}")
            os.makedirs(self.seg_out_dir)

        if not os.path.exists(self.det_out_dir):
            print(f"creating dir: {self.det_out_dir}")
            os.makedirs(self.det_out_dir)

        if not os.path.exists(self.output_tils_dir):
            print(f"creating dir: {self.output_tils_dir}")
            os.makedirs(self.output_tils_dir)
