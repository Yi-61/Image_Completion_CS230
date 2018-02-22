from PIL import Image
import numpy as np
import os, os.path
import pickle

folder_path = '/Users/yiliu/Box Sync/eclipse-workspace/CS230_Image_Completion/Subset/'
pickle_path = folder_path + 'data.pickle'
data_read = np.load( pickle_path )
print(data_read.shape)

#For test
for i in range(100):
    img_test = Image.fromarray( data_read[i, :, :, :].astype("uint8"), 'RGB')
    img_test.save( str(i) + '_processed.jpg' )