import os

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "runs")


class Config:
    def __init__(
        self,
        wsi_dir="/home/u1910100/Documents/Tiger_Data/wsitils/images",
        output_dir="/home/u1910100/Documents/Tiger_Data/prediction",
        model_dir=DEFAULT_MODEL_DIR,
    ) -> None:
        self.output_dir = output_dir
        self.wsi_dir = wsi_dir

        self.temp_out_dir = os.path.join(self.output_dir, "temp_out/")
        self.seg_out_dir = os.path.join(self.output_dir, "seg_out_v2/")
        self.det_out_dir = os.path.join(self.output_dir, "det_out_v2/")
        self.output_tils_dir = os.path.join(self.output_dir, f"tils_v2/")
        # self.create_dirs()

        self.model_dir = model_dir
        self.cell_model_dir = os.path.join(self.model_dir, "cell/weights")
        self.tissue_model_dir = os.path.join(self.model_dir, "tissue/weights")

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
        wsi_dir="/input",
        output_dir="/output/",
        model_dir=DEFAULT_MODEL_DIR,
    ) -> None:
        self.output_dir = output_dir
        self.wsi_dir = wsi_dir

        self.temp_out_dir = os.path.join(self.wsi_dir, "images/")  # Input mask dir
        self.seg_out_dir = os.path.join(
            self.output_dir, "images/breast-cancer-segmentation-for-tils/"
        )
        self.det_out_dir = self.output_dir
        self.output_tils_dir = self.output_dir
        # self.create_dirs()

        self.model_dir = model_dir
        self.cell_model_dir = os.path.join(self.model_dir, "cell/weights")
        self.tissue_model_dir = os.path.join(self.model_dir, "tissue/weights")

    def create_dirs(self):
        if not os.path.exists(self.temp_out_dir):
            os.makedirs(self.temp_out_dir)
        if not os.path.exists(self.seg_out_dir):
            os.makedirs(self.seg_out_dir)
        if not os.path.exists(self.det_out_dir):
            os.makedirs(self.det_out_dir)
        if not os.path.exists(self.output_tils_dir):
            os.makedirs(self.output_tils_dir)


DefaultConfig = Config()
ChallengeConfig = Challenge_Config()
