import logging
import abc
import os

from osgeo import ogr, gdal, gdal_array

class DataImporter:
    """Represents a toolset for importing the necessary data for an image classification.
    """

    def __init__(self):
        
        gdal.UseExceptions()
        gdal.AllRegister()

    def read_img_data(self, img_path):
        """Reads the image data, that should be classified using gdal.

        .. table::
            :widths: auto

            ================    ====================================================
            Argument            Description
            ----------------    ----------------------------------------------------
            img_path            String. Path to image including filename.
            ================    ====================================================

        :return: gdal object -- The imported image data, None if failed.
        """
        logging.debug("Reading image data...")
        if img_path:
            try:
                img_ds = gdal.Open(img_path, gdal.GA_ReadOnly)
                logging.debug("Successfully opened image data.")
                return img_ds
            except Exception as e:
                logging.error('Failed to read image data: {e}'.format(e=e))
                return None
        else:
            logging.error('No image path specified.')
            return None
    
    def read_training_samples(self, sample_path):
        """Returns the trainng sample data. Data is opened using ogr and sample infos are checked.

        Following sample data types are supported: ESRI shapefile

        .. table::
            :widths: auto

            ================    =========================================================
            Argument            Description
            ----------------    ---------------------------------------------------------
            sample_path         String. Path to training sample data including filename.
            ================    =========================================================  

        :return: ogr object -- =The imported training sample data. None if failed.     
        """
        logging.debug('Reading traning sample data.')
        if sample_path:
            try:
                dataset = ogr.Open(sample_path)
                if not dataset:
                    logging.error('Could not open training data.')
            except Exception as e:
                logging.error('Failed to open training samples: {e}'.format(e=e))
                return None

            # getting dshapefile infos
            driver = dataset.GetDriver()
            layer_count = dataset.GetLayerCount()
            layer = dataset.GetLayerByIndex(0)
            geometry = layer.GetGeomType()
            geometry_name = ogr.GeometryTypeToName(geometry)
            spatial_ref = layer.GetSpatialRef()
            proj4 = spatial_ref.ExportToProj4()
            feature_count = layer.GetFeatureCount()
            defn = layer.GetLayerDefn()
            field_count = defn.GetFieldCount()
            fields_list = []
            for i in range(field_count):
                field_defn = defn.GetFieldDefn(i)
                fields_list.append(field_defn.GetName())

            logging.info("Sample data loaded: ")
            logging.info("")
            logging.info("+--------------------+--------------------------------------+")
            logging.info("| Dataset Driver     | {d}".format(d=driver.name))
            logging.info("+--------------------+--------------------------------------+")
            logging.info("| Number of Layers   | {n}".format(n=layer_count))
            logging.info("+--------------------+--------------------------------------+")
            logging.info("| First Layer Name   | {f}".format(f=layer.GetName()))
            logging.info("+--------------------+--------------------------------------+")
            logging.info("| Layer geometry     | {l}".format(l=geometry_name))
            logging.info("+--------------------+--------------------------------------+")
            logging.info("| Spatial referene   | {p}".format(p=proj4))
            logging.info("+--------------------+--------------------------------------+")
            logging.info("| Feature count      | {c}".format(c=feature_count))
            logging.info("+--------------------+--------------------------------------+")
            logging.info("| Field names        | {f}".format(f=str(fields_list)))
            logging.info("+--------------------+--------------------------------------+")
            logging.info("")

            return dataset

        else:
            logging.error("No sample path specified.")
            return None