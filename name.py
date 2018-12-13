from __future__ import print_function, division
from osgeo import ogr, gdal, gdal_array

import os
import shutil
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.externals import joblib

# set directories
# rootdir of image data
path_image = "/home/sandro/Documents/EAGLE_Data/WS201819_1st_Term/04GEOMB1_Digital_Image_Analysis/Steigerwald/03_raster/01_landsat/02_timescan"
if not os.path.exists(path_image):
    print("ERROR: Image root directory not found.")
else:
    print("INFO: rootdir set")

# path to training data shapefile
path_train = "/home/sandro/Documents/EAGLE_Data/WS201819_1st_Term/04GEOMB1_Digital_Image_Analysis/Steigerwald/GIS_Analysis"
if not os.path.exists(path_train):
    print("ERROR: Training sample root directory not found.")
else:
    print("INFO: path_train set")

# path to classification model
path_model = "/home/sandro/Documents/EAGLE_Data/WS201819_1st_Term/04GEOMB1_Digital_Image_Analysis/Class_Py/model"
if not os.path.exists(path_model):
    print("ERROR: Model root directory not found.")
else:
    print("INFO: path_model set")

# path to classification output
path_class = "/home/sandro/Documents/EAGLE_Data/WS201819_1st_Term/04GEOMB1_Digital_Image_Analysis/Class_Py/output"
if not os.path.exists(path_class):
    print("ERROR: class root directory not found.")
else:
    print("INFO: path_class set")


# -------- read arcgis training data ---------


dataset = ogr.Open(os.path.join(path_train, 'training_samples_Arcgis.shp'))
if not dataset:
    print('Error: could not open training data')

### Let's get the driver from this file
driver = dataset.GetDriver()
print('Dataset driver is: {n}\n'.format(n=driver.name))

### How many layers are contained in this Shapefile?
layer_count = dataset.GetLayerCount()
print('The shapefile has {n} layer(s)\n'.format(n=layer_count))

### What is the name of the 1 layer?
layer = dataset.GetLayerByIndex(0)
print('The layer is named: {n}\n'.format(n=layer.GetName()))

### What is the layer's geometry? is it a point? a polyline? a polygon?
# First read in the geometry - but this is the enumerated type's value
geometry = layer.GetGeomType()

# So we need to translate it to the name of the enum
geometry_name = ogr.GeometryTypeToName(geometry)
print("The layer's geometry is: {geom}\n".format(geom=geometry_name))

### What is the layer's projection?
# Get the spatial reference
spatial_ref = layer.GetSpatialRef()

# Export this spatial reference to something we can read... like the Proj4
proj4 = spatial_ref.ExportToProj4()
print('Layer projection is: {proj4}\n'.format(proj4=proj4))

### How many features are in the layer?
feature_count = layer.GetFeatureCount()
print('Layer has {n} features\n'.format(n=feature_count))

### How many fields are in the shapefile, and what are their names?
# First we need to capture the layer definition
defn = layer.GetLayerDefn()

# How many fields
field_count = defn.GetFieldCount()
print('Layer has {n} fields'.format(n=field_count))

# What are their names?
print('Their names are: ')
for i in range(field_count):
    field_defn = defn.GetFieldDefn(i)
    print('\t{name} - {datatype}'.format(name=field_defn.GetName(),
                                         datatype=field_defn.GetTypeName()))

# --------transform Training data-------------
from osgeo import gdal

# open raster image
raster_ds = gdal.Open(os.path.join(path_image, 'TimeScan_EAGLE_AOI.tif'), gdal.GA_ReadOnly)

# fetch number of# Fetch number of rows and columns
ncol = raster_ds.RasterXSize
nrow = raster_ds.RasterYSize

# Fetch projection and extent
proj = raster_ds.GetProjectionRef()
ext = raster_ds.GetGeoTransform()

raster_ds = None

# Create the raster dataset
memory_driver = gdal.GetDriverByName('GTiff')
out_raster_ds = memory_driver.Create(os.path.join(path_train, 'training_data.gtif'), ncol, nrow, 1, gdal.GDT_Byte)

# Set the ROI image's projection and extent to our input raster's projection and extent
out_raster_ds.SetProjection(proj)
out_raster_ds.SetGeoTransform(ext)

# Fill our output band with the 0 blank, no class label, value
b = out_raster_ds.GetRasterBand(1)
b.Fill(0)

status = gdal.RasterizeLayer(out_raster_ds,  # output to our new dataset
                             [1],  # output to our new dataset's first band
                             layer,  # rasterize this layer
                             None, None,  # don't worry about transformations since we're in same projection
                             [0],  # burn value 0
                             ['ALL_TOUCHED=TRUE',  # rasterize all pixels touched by polygons
                              'ATTRIBUTE=Classvalue']  # put raster values according to the 'id' field values
                             )

# Close dataset
out_raster_ds = None

if status != 0:
    print("Error.")
else:
    print("Success")



#---------check training raster ----------------
# Import NumPy for some statistics
import numpy as np

roi_ds = gdal.Open(os.path.join(path_train, 'training_data.gtif'), gdal.GA_ReadOnly)

roi = roi_ds.GetRasterBand(1).ReadAsArray()

# How many pixels are in each class?
classes = np.unique(roi)
# Iterate over all class labels in the ROI image, printing out some information
for c in classes:
    print('Class {c} contains {n} pixels'.format(c=c,
                                                 n=(roi == c).sum()))



#------- opening the images -------------
gdal.UseExceptions()
gdal.AllRegister()

# read image and ROI
img_ds = gdal.Open(os.path.join(path_image, 'TimeScan_EAGLE_AOI.tif'), gdal.GA_ReadOnly)
roi_ds = gdal.Open(os.path.join(path_train, 'training_data.gtif'), gdal.GA_ReadOnly)

img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
                gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
for b in range(img.shape[2]):
    img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

roi = roi_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)

# display them
"""
plt.subplot(121)
plt.imshow(img[:,:,4], cmap=plt.cm.Greys_r)
plt.title('Band 4')

plt.subplot(122)
plt.imshow(roi, cmap=plt.cm.Spectral)
plt.title('ROI Training Data')

plt.show()
"""

# --------- pairing image and training --------
# find how many training samples
n_samples = (roi > 0).sum()
print('We have {n} samples'.format(n=n_samples))

# classification labels
labels = np.unique(roi[roi > 0])
print('Training data include {n} classes: {classes}'.format(n=labels.size, classes=labels))

X = img[roi > 0, :]
y = roi[roi>0]

print('X martix is sized: {sz}'.format(sz=X.shape))
print('Y matrix is sized: {sz}'.format(sz=y.shape))


# train the Classifier
rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=4,
                            min_samples_split=2, min_samples_leaf=1, max_features='auto',
                            bootstrap=True, oob_score=True, n_jobs=1, random_state=None, verbose=True)
rf = rf.fit(X,y)

print('OOB prediciton of accuracy is: {oob}%'.format(oob=rf.oob_score_ * 100))

# export Random Forest Model
model = os.path.join(path_model, 'model_rf.pkl')
joblib.dump(rf, model)

new_shape = (img.shape[0] * img.shape[1], img.shape[2])
img_as_array = img[:,:,:].reshape(new_shape)
print('Reshaped from {o} to {n}'.format(o=img.shape, n=img_as_array.shape))

class_prediction = rf.predict(img_as_array)

class_prediction = class_prediction.reshape(img[:,:,0].shape)

# export classification
classification = os.path.join(path_class, 'classification.tif')
io.imsave(classification, class_prediction)



def training():

    # path to image
    raster = os.path.join(path_image, 'TimeScan_EAGLE_AOI.tif')
    # path to training
    samples = os.path.join(path_train, 'training_data.gtif')

    # read image
    img_ds = io.imread(raster)
    # convert to 16bit numpy array
    img = np.array(img_ds, dtype='int16')

    # same with sample pixels
    roi_ds = io.imread(samples)
    roi = np.array(roi_ds, dtype='int8')

    # read the labels
    labels = np.unique(roi[roi > 0])
    print('Training data include {n} classes: {classes}'.format(n=labels.size, classes = labels))

    # compose X,Y data (dataset - training data)
    X = img[roi >0, :]
    Y = roi[roi > 0]

    # assign class weights
    # weights = {1:3, 2:2, 3:2, 4:2} # etc

    # build random forest classifier
    rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=4,
                                min_samples_split=2, min_samples_leaf=1, max_features='auto',
                                bootstrap=True, oob_score=True, n_jobs=1, random_state=None, verbose=True)
    
    # fit training data with original dataset
    rf = rf.fit(X,Y)

    # export Random FOrest Model
    model = os.path.join(path_model, 'model_rf.pkl')
    joblib.dump(rf, model)

def classify():

    # path to image
    raster = os.path.join(path_image, 'TimeScan_EAGLE_AOI.tif')
    
    # read data
    img_ds = io.imread(raster)
    img = np.array(img_ds, dtype='int16')

    # call random forest model
    rf = os.path.join(path_model, 'model_rf.pkl')
    clf = joblib.load(rf)

    # actual classification
    new_shape = (img.shape[0]*img.shape[1], img.shape[2])
    img_as_array = img[:,:,:23].reshape(new_shape)


    class_predicition = clf.predict(img_as_array)
    class_predicition = class_predicition.reshape(img[:,:,0].shape)

    # export classification
    classification = os.path.join(path_class, 'classification.tif')
    io.imsave(classification, class_predicition)



#training()
#classify()