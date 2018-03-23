from PIL import Image
import numpy as np
import os.path
import pickle
import time

# Read landmarks
# Delete the first two rows of 'list_landmarks_align_celeba.txt' and rename it to 'landmarks_align.txt'
def read_landmarks(landmark_folder_path):
    landmarks_path = os.path.join(landmark_folder_path, 'landmarks_align.txt')
    landmarks_file = open(landmarks_path, 'r')
    landmarks = np.loadtxt(landmarks_file, usecols = (1,2,3,4,5,6,7,8,9,10))
    return landmarks

# Return an image
def read_one_image(image_folder_path, image_name):
    image_path = os.path.join(image_folder_path, image_name)
    image = Image.open(image_path)
    return image

# Crop one image and resize it
# Return ndarray
def resize_one_image(index, image, landmarks, save_flag = False, save_folder_path = './celeba_data/image_align_processed'):
    image_data = np.asarray(image, dtype = "int32")
    width, height = image_data.shape[0], image_data.shape[1]
    left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, nose_y, left_mouth_x, left_mouth_y, right_mouth_x, right_mouth_y = landmarks[index-1]
    length_crop = 80
    edge_to_eye = 20
    edge_to_mouth = 60
    length_resize = 80
    if left_eye_x - edge_to_eye < 0:
        left = 0
        right = left + length_crop
    elif right_eye_x + edge_to_eye > width - 1:
        right = width - 1
        left = right - length_crop
    else:
        left = left_eye_x - edge_to_eye
        right = left + length_crop

    mouth_mean_y = np.mean([left_mouth_y, right_mouth_y])
    if mouth_mean_y - edge_to_mouth < 0:
        upper = 0
        lower = upper + length_crop
    elif mouth_mean_y + (length_crop - edge_to_mouth) > height - 1:
        lower = height - 1
        upper = lower - length_crop
    else:
        upper = mouth_mean_y - edge_to_mouth
        lower = upper + length_crop

    image_cropped = image.crop((left, upper, right, lower))
    image_resized = image_cropped.resize((length_resize, length_resize))

    if save_flag:
        resized_image_name = 'resized_' + str(length_resize) + '_' + str(index).zfill(6) + '.jpg'
        image_resized.save(os.path.join(save_folder_path, resized_image_name))

    return image_resized

# Write array of image data to pickle files
def write_to_pickle(landmark_folder_path = './celeba_data', image_folder_path = './celeba_data/image_align',
        save_folder_path = './celeba_data/pickle', start_index = 1, end_index = 202599, verbose_step = 1000, save_step = 10000):
    if start_index < 1:
        start_index = 1
    if end_index > 202599:
        end_index = 202599

    image_data_list = []
    landmarks = read_landmarks(landmark_folder_path)
    print('Landmarks read')

    last_saved_index = start_index - 1
    tic = time.time()
    for index in range(start_index, end_index + 1):
        image_name = str(index).zfill(6) + '.jpg'
        image = read_one_image(image_folder_path, image_name)
        image_resized = resize_one_image(index, image, landmarks)
        image_resized_data = np.asarray(image_resized, dtype='uint8')
        image_data_list.append(image_resized_data)

        if index % verbose_step == 0:
            print('Completed image index: ' + str(index))

        if index % save_step == 0 or index == end_index:
            image_data_array = np.array(image_data_list)

            pickle_name = str(last_saved_index + 1).zfill(6) + '_' + str(min(end_index, index)).zfill(6) +'.pickle'
            pickle_path = os.path.join(save_folder_path, pickle_name)
            pickle.dump(image_data_array, open(pickle_path,'wb'))

            toc = time.time()
            print('Saved to file: ' + pickle_name)
            print('Time for this batch is: ' + str(toc - tic) + 's')

            image_data_list = []
            last_saved_index = index
            tic = time.time()

# Run this line
write_to_pickle()
