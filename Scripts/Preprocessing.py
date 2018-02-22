from PIL import Image
import numpy as np
import os, os.path
import pickle

folder_path = '/Users/yiliu/Box Sync/eclipse-workspace/CS230_Image_Completion/Subset/'

#read landmarks
#delete the first two rows of 'list_landmarks_align_celeba.txt' and rename it to 'list_landmarks_align_celeba_new.txt'
def read_landmarks( folder_path ):
    landmarks_path = folder_path + 'Anno/list_landmarks_align_celeba_new.txt'
    landmarks_file = open( landmarks_path, 'r' )
    landmarks = np.loadtxt(landmarks_file, usecols = (1,2,3,4,5,6,7,8,9,10) )
    return landmarks

#return the img object
def read_one_image( folder_path, image_name ):
    image_path = folder_path + image_name
    img = Image.open(image_path)
    img.load()
    return img

#crop one image to 120*120 pixels, and resize it
#return ndarray
def resize_one_image( i, img, image_name, landmarks, folder_path ):
    image_data = np.asarray(img, dtype = "int32")
    width, height = image_data.shape[0], image_data.shape[1]
    [lefteye_x, lefteye_y, righteye_x, righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y] = landmarks[i]
    length_crop = 110
    edge2eye = 35
    edge2mouth = 85
    legnth_resize = 64
    if lefteye_x - edge2eye < 0:
        left = 0
        right = left + length_crop
    elif righteye_x + edge2eye > width - 1:
        right = width - 1
        left = right - length_crop
    else:
        left = lefteye_x - edge2eye
        right = left + length_crop
    
    mouth_mean_y = np.mean([leftmouth_y, rightmouth_y])
    if mouth_mean_y - edge2mouth < 0:
        upper = 0
        lower = upper + length_crop
    elif mouth_mean_y + (length_crop - edge2mouth) > height - 1:
        lower = height - 1
        upper = lower - length_crop
    else:
        upper = mouth_mean_y - edge2mouth
        lower = upper + length_crop
    
    image_crop = img.crop((left, upper, right, lower))
    image_resize = image_crop.resize( (legnth_resize, legnth_resize) )
    image_resize = np.asarray(image_resize, dtype = "int32")
    img_resize = Image.fromarray( image_resize.astype("uint8"), "RGB")
    img_resize.save( folder_path + 'Processed/' + os.path.splitext(image_name)[0] + '_resize.jpg')
    return image_resize

#append a new image(64*64*3) to data(m*64*64*3) (nparray)
def append_image(data, image, resize = 64):
    new_image = image.reshape( (1, resize, resize, 3) )
    data = np.append( data, new_image, axis = 0 )
    return data

#combine all images in a folder into an ndarray 'data'
def create_data( folder_path, valid_images = [".jpg"], resize = 64 ):
    data = np.zeros( (1, resize, resize, 3) )
    landmarks = read_landmarks( folder_path )
    for image_name in os.listdir( folder_path ):
        ext = os.path.splitext(image_name)[1]
        if ext.lower() not in valid_images:
            continue
        img = read_one_image( folder_path, image_name )
        i = int( image_name[0] ) - 1 #the No. of image
        image_resize = resize_one_image( i, img, image_name, landmarks, folder_path )
        data = append_image( data, image_resize )
    data = np.delete( data, 0, axis = 0)
#     for i in range(5):
#         img_test = Image.fromarray( data[i, :, :, :].astype("uint8"), 'RGB')
#         img_test.save(str(i) + '_test.jpg')
    return data

data = create_data(folder_path)
print(data.shape)
pickle_path = folder_path + 'data.pickle'
pickle.dump( data, open( pickle_path,'ab') )
# data.dump( pickle_name )
# data_read = np.load( pickle_path )
# print(data_read.shape)


