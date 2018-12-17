import logging
import os

from osgeo import ogr, gdal, gdal_array

class PyImageClassifier:

    def __init__(self, img_path=None, sample_path=None, model_path=None, out_path=None):
        
        self.img_path       = img_path
        self.sample_path    = sample_path
        self.model_path     = model_path
        self.out_path       = out_path

        self.samples        = self.read_training_samples()

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

    def read_training_samples(self):

        logging.debug('Reading traning sample data.')
        if self.sample_path:
            try:
                dataset = ogr.Open(self.sample_path)
                if not dataset:
                    logging.error('Could not open training data.')
            except Exception as e:
                logging.error('Failed to open training samples: {e}'.format(e=e))
                return None

            # getting dataset driver
            driver = dataset.GetDriver()
            logging.debug('Dataset driver is: {n}'.format(n=driver.name))

            # checking layers in dataset
            layer_count = dataset.GetLayerCount()
            logging.debug('Number of layers in shapefile: {n}'.format(n=layer_count))

            # getting name of the 1. layer
            layer = dataset.GetLayerByIndex(0)
            logging.debug('First layer name: {n}'.format(n=layer.GetName()))

            # getting geometry info
            geometry = layer.GetGeomType()
            geometry_name = ogr.GeometryTypeToName(geometry)
            logging.debug("Layer geometry: {geom}".format(geom=geometry_name))

            # getting spatial reference info
            spatial_ref = layer.GetSpatialRef()
            proj4 = spatial_ref.ExportToProj4()
            logging.debug('Spatial reference: {proj4}'.format(proj4=proj4))

            # getting feature count
            feature_count = layer.GetFeatureCount()
            logging.debug('Feature count: {n}'.format(n=feature_count))

            # getting field count
            defn = layer.GetLayerDefn()
            field_count = defn.GetFieldCount()
            logging.debug('Field count: {n}'.format(n=field_count))

            # getting field names
            for i in range(field_count):
                field_defn = defn.GetFieldDefn(i)
                logging.debug('\t{name} - {datatype}'.format(name=field_defn.GetName(),
                                                            datatype=field_defn.GetTypeName()))
        else:
            logging.error("No sample path specified.")
            return None