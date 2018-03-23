from PIL import Image
import numpy as np
import pickle
import os.path

def load(pickle_name, folder_path = './celeba_data/'):
    pickle_path = os.path.join(folder_path, pickle_name)
    data_read = np.load(pickle_path)
    return data_read

'''
# For test purposes
for i in range(100):
    img_test = Image.fromarray(data_read[i, :, :, :].astype("uint8"), 'RGB')
    img_test.save(str(i) + '_processed.jpg')
'''
