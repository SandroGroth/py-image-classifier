import logging
import abc
import os

from osgeo import ogr, gdal, gdal_array

class DataImporter:

    def __init__(self):
        
        gdal.UseExceptions()
        gdal.AllRegister()

    def read_training_samples(self, sample_path):

        logging.debug('Reading traning sample data.')
        if sample_path:
            try:
                dataset = ogr.Open(sample_path)
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

            return dataset

        else:
            logging.error("No sample path specified.")
            return None