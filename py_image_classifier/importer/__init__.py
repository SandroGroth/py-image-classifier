import logging
import os

from osgeo import ogr, gdal, gdal_array

class DataImporter:

    def __init__(self):
        
        gdal.UseExceptions()
        gdal.AllRegister()

    