import os


class Config:
    def __init__(
        self,
        wsi_dir="/home/u1910100/Documents/Tiger_Data/wsitils/images",
        output_dir="/home/u1910100/Documents/Tiger_Data/prediction",
    ) -> None:
        self.output_dir = output_dir
        self.wsi_dir = wsi_dir

        self.temp_out_dir = os.path.join(self.output_dir, "temp_out/")
        self.seg_out_dir = os.path.join(self.output_dir, "seg_out/")
        self.det_out_dir = os.path.join(self.output_dir, "det_out/")
        self.output_tils_dir = os.path.join(self.output_dir, f"tils/")
        self.create_dirs()

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
