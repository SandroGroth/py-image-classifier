import logging
import os

class PyImageClassifier:

    def __init__(self, img_path=None, sample_path=None, model_path=None, out_path=None):
        
        self.img_path = img_path
        self.sample_path = sample_path
        self.model_path = model_path
        self.out_path = out_path

    def set_img_path(self, img_path):
        
        if os.path.exists(img_path):
            self.img_path = img_path
        else:
            logging.error("Specified image path does not exist.")

        return

    def set_sample_path(self, sample_path):
        
        if os.path.exists(sample_path):
            self.sample_path = sample_path
        else:
            logging.error("Specified training sample path does not exist.")

        return

    def set_model_path(self, model_path):
        
        if os.path.exists(model_path):
            self.model_path = model_path
        else:
            logging.error("Specified model path does not exist.")

        return

    def set_out_path(self, out_path):
        
        if os.path.exists(out_path):
            self.out_path = out_path
        else:
            logging.error("Specified classification output path does not exist.")

        return