import logging
from py_image_classifier import PyImageClassifier

logging.basicConfig(level=logging.DEBUG)

img_path = "/home/sandro/Documents/EAGLE_Data/WS201819_1st_Term/04GEOMB1_Digital_Image_Analysis/Steigerwald/03_raster/01_landsat/02_timescan/TimeScan_EAGLE_AOI.tif"
sample_path = "/home/sandro/Documents/EAGLE_Data/WS201819_1st_Term/04GEOMB1_Digital_Image_Analysis/Steigerwald/GIS_Analysis/training_samples_Arcgis.shp"
model_path = "/home/sandro/Documents/EAGLE_Data/WS201819_1st_Term/04GEOMB1_Digital_Image_Analysis/Class_Py/model"
out_path = "/home/sandro/Documents/EAGLE_Data/WS201819_1st_Term/04GEOMB1_Digital_Image_Analysis/Class_Py/output"

py_image_classifier = PyImageClassifier(img_path, sample_path, model_path, out_path)

# for debugging purposes
print("")