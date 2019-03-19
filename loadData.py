
import pandas as pd
import numpy as np

# We define an array to represent the facies adjacent to each other.  For facies label `i`, `adjacent_facies[i]` is an 
# array of the adjacent facies labels.
#adjacent_facies = np.array([[1], [0,2], [1], [4], [3,5], [4,6,7], [5,7], [5,6,8], [6,7]])

# 1=sandstone  2=c_siltstone   3=f_siltstone 
# 4=marine_silt_shale 5=mudstone 6=wackestone 7=dolomite
# 8=packstone 9=bafflestone

seam_colors = ['#FFFFFF','#FFFF00','#1CE6FF','#FF34FF','#FF4A46','#008941','#006FA6','#A30059','#FFDBE5','#7A4900','#0000A6','#63FFAC','#B79762','#004D43','#8FB0FF','#997D87','#5A0007','#809693','#FEFFE6','#1B4400','#4FC601','#3B5DFF','#4A3B53','#FF2F80','#61615A','#BA0900','#6B7900','#00C2A0','#FFAA92','#FF90C9','#B903AA','#D16100']
seam_labels= ['None','10A','10B','10R','2L','2U','3L1','3L2','3LA','3UA','3UB','4JG','4L','4R','4U','5M','6L','6U','7L','7R','8M','8R','9L','A','B','C','CO','D','GA','K4','R4U','R6L']

#seams_color_map is a dictionary that maps seam labels to their respective colors
seams_color_map = {}
for ind, seamLabel in enumerate(seam_labels):
    seams_color_map[seamLabel] = seam_colors[ind]

def fillNoneSeam(data):
    noSeamDataIndexes = data[pd.isnull(data['ASEAM'])].index
    data.loc[noSeamDataIndexes,'ASEAM'] = 'None'
    return data

def removeEmptyValues(data):
    notEmptyRASH = data['RASH'].notnull().values
    data = data[notEmptyRASH]
    notEmptyRFSI = data['RFSI'].notnull().values
    data = data[notEmptyRFSI]
    return data

def get_seam_index(row):
    seamLabel = row['ASEAM']
    seamIndex = seam_labels.index(seamLabel)
    return seamIndex

def appendSeamIndex(data):
    data['SEAM'] = data.apply(lambda row: get_seam_index(row), axis=1)
    return data

def loadAndPrepareData():
    filename = 'csamp-assay-seams.csv'
    training_data = pd.read_csv(filename)
    training_data = fillNoneSeam(training_data)
    #training_data = removeEmptyValues(training_data)
    training_data = appendSeamIndex(training_data)

    training_data['DHID'] = training_data['DHID'].astype('category')
    
    blind = training_data[training_data['DHID'] == 'D428']
    training_data = training_data[training_data['DHID'] != 'D428']

    return training_data, blind