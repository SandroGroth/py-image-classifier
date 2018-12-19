import logging
import os

from py_image_classifier.importer import DataImporter
from osgeo import ogr, gdal, gdal_array

class PyImageClassifier:

    def __init__(self, img_path=None, sample_path=None, model_path=None, out_path=None):
        
        self.img_path       = img_path
        self.sample_path    = sample_path
        self.model_path     = model_path
        self.out_path       = out_path

        self.importer       = DataImporter()
        #self.img            = self.importer.read_img_data(self.img_path)
        self.samples        = self.importer.read_training_samples(self.sample_path)

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

    def read_img_data(self):

        logging.debug("Reading image data...")
        if self.img_path:
            try:
                img_ds = gdal.Open(self.img_path, gdal.GA_ReadOnly)
                logging.debug("Successfully opened image data.")
                return img_ds
            except Exception as e:
                logging.error('Failed to read image data: {e}'.format(e=e))
                return None
        else:
            logging.error('No image path specified.')
            return None