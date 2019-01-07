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
        self.img            = self.importer.read_img_data(self.img_path)
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

    def rasterize_samples(self, attribute_name, out_path, file_name='training_raster'):
        """
        memory_driver == 'GTiff'
        """
        if not self.img:
            logging.error("Iamge has to be imported.")
            return None

        # fetch resolution of img
        ncol = self.img.RasterXSize
        nrow = self.img.RasterYSize

        # fetch extent and projection of img
        proj = self.img.GetProjectionRef()
        ext = self.img.GetGeoTransform()

        # get the information layer
        layer = self.samples.GetLayerByIndex(0)

        # Create the raster dataset
        driver = gdal.GetDriverByName('GTiff')
        out_raster_ds = driver.Create(os.path.join(out_path, file_name + '.gtif'), ncol, nrow, 1, gdal.GDT_Byte)

        # set same extent and projection as img
        out_raster_ds.SetProjection(proj)
        out_raster_ds.SetGeoTransform(ext)

        # Fill our output band with the 0 blank, no class label, value
        b = out_raster_ds.GetRasterBand(1)
        b.Fill(0)

        status = (out_raster_ds,  # output to our new dataset
                    [1],  # output to our new dataset's first band
                    layer,  # rasterize this layer
                    None, None,  # don't worry about transformations since we're in same projection
                    [0],  # burn value 0
                    ['ALL_TOUCHED=TRUE',  # rasterize all pixels touched by polygons
                    'ATTRIBUTE={}'.format(attribute_name)]  # put raster values according to the 'id' field values
                    )

        return status