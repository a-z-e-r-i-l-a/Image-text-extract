import os
import sys
from io import StringIO
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.pylab import savefig
import pytesseract
import cv2
from skimage.filters import threshold_niblack, threshold_mean
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from PIL import Image
import pickle
import scipy.io


def errosion_and_extrapolation(image_name=None, image_array=None, output_name='Cleaned_Image.jpg', plot=1):
    """
    Given an image, tries to remove vertical/horizontal borders

    :param image_name: path+name of the image
    :param image_array: 2D image np.array (if image_name is None)
    :param output_name: name to be saved (if output_name is not None )
    :param plot: whether to show a plot of the erroded image or not
    :return: erroded image array,
    """
    if image_array is None:
        img_cut = cv2.imread(image_name, 0)
    else:
        img_cut = image_array

    if plot:
        plt.figure(figsize=(35, 50))
        plt.imshow(img_cut)
        plt.title('original image')
    if image_array is None:
        blur = cv2.GaussianBlur(img_cut, (1, 1), 0)[:, :, 0]
    else:
        blur = cv2.GaussianBlur(img_cut, (1, 1), 0)
    otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img_inverted = cv2.bitwise_not(otsu)

    # dilation/erosion horizontal lines
    horizontal = cv2.erode(img_inverted, cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1)))
    horizontal = cv2.dilate(horizontal, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1)))

    # dilation/erosion vertical lines
    vertical = cv2.erode(img_inverted, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60)))
    vertical = cv2.dilate(vertical, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10)))

    mask = cv2.bitwise_not(horizontal + vertical)
    img_cut_mask = cv2.bitwise_not(cv2.bitwise_and(img_inverted, mask))
    if plot:
        plt.figure(figsize=(35, 50))
        plt.imshow(img_cut_mask)
        plt.title('erroded image')

    bw = np.array(closing(img_cut_mask > 10) * 255, dtype=np.uint8)

    if img_cut_mask is None:
        img_cut_mask = cv2.erode(img_cut, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))
    else:
        img_cut_mask = cv2.erode(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))

    if plot:
        plt.figure(figsize=(35, 50))
        plt.imshow(img_cut_mask)
        plt.title('erroded and extrapolated image')

    if output_name is not None:
        Image.fromarray(img_cut_mask).save(output_name)
    return img_cut_mask


def scikitting(image_array,
               black_region=True,
               area_threshold=5000,
               thresholding='mean',
               clearborder=1,
               minDist=40,
               make_img_label_overlay=1,
               unite_boxes=1):
    """
    Given an image array, finds the closed black or white regions
    Used for masking words(finding coordinates of the regions that include a word inside them)

    :param image_array: 2D image np.array
    :param black_region: To consider loops around black regions or white regions for labeling
    :param area_threshold: Minimum area of the labeled region to be accepted
    :param thresholding: Scikit image thresholding method to be used for the image. choices: 'mean', niblack
    :param clearborder: Whether to clear the borders of the image before labeling
    :param minDist: Minumum distance required for two regions to merge (if unite_boxes is True)
    :param make_img_label_overlay: Makes a colored image array showing the labels
    :param unite_boxes: After labeling the the regions in image, unites close/overlapping ones
    :return: Pandas DataFrame of regions' coordinates
    """
    if thresholding == 'mean':
        thresh = threshold_mean(image_array)
    else:
        thresh = threshold_niblack(image_array)
    if black_region:
        bw = closing(image_array < thresh)
    else:
        bw = closing(image_array > thresh)
    if clearborder:
        cleared = clear_border(bw)
    else:
        cleared = bw
    label_image = label(cleared)
    if make_img_label_overlay:
        image_label_overlay = label2rgb(label_image, image=image_array / 255)
    else:
        image_label_overlay = None
    labels = [
                'top',
                'left',
                'Y_coordinate2',
                'X_coordinate2',
                'row_id',
                'group_id',
                'column_id',
                'upper_label',
                'left_label'
            ]

    column_id = None
    group_id = None
    row_id = None
    column_id = None
    upper_label = None
    left_label = None
    table_list = []
    for region in regionprops(label_image):
        if region.area >= area_threshold:
            top, left, Y_coordinate2, X_coordinate2 = region.bbox
            width = X_coordinate2 - left
            height = Y_coordinate2 - top
            aspect_ratio = width / height

            if aspect_ratio < 27 and aspect_ratio > 1 / 10:
                table_list.append(list(region.bbox)
                                  + [row_id]
                                  + [group_id]
                                  + [column_id]
                                  + [upper_label]
                                  + [left_label]
                                  )
    if unite_boxes:
        table_list = unite_close_boxes(table_list, minDist=minDist)
        return pd.DataFrame(table_list, columns=labels), image_label_overlay
    else:
        return pd.DataFrame(table_list, columns=labels), image_label_overlay


def unite_close_boxes(table_list, minDist=50):
    """
    Merges close or overlapping regions.

    :param table_list: A list of the following format which includes the coordinates of the regions
             [
             [top , left, Y_coordinate2, X_coordinate2],
             ...]
    :param minDist: Minimum distance for considering two regions to be close and merged
    :return: A list with the same format of the input which includes coordinates of the merged regions
    """

    boxes_found_to_be_merged = 1
    row_id = 1
    while boxes_found_to_be_merged:
        boxes_found_to_be_merged = 0
        for index1 in range(len(table_list)):  # Picks the index of one of the masks (with index1 in the list)
            for index2 in range(len(table_list)):   # Every other mask (with index2 in the list) it checks the
                                                    # following conditions.
                if (index1 != index2    # Making sure we are picking a mask other than the one with index1
                    and len(table_list) > index1        # Since we might have deleted a mask in extend_box_and_drop
                        and len(table_list) > index2):  # with an index in the list in some previous iterations of
                                                        # this while loop, we make sure index 1 and tow does not
                                                        # exceed the length of the table_list
                    if table_list[index1][4] is None:       # checks if a row_id (4th column) is not assigned to
                                                            # this mask before it otherwise assigns a row_id to it
                        table_list[index1][4] = row_id
                        row_id += 1

                    if is_in_one_row(table_list, index1, index2):
                        table_list[index2][4] = table_list[index1][4]
                        if (are_close(table_list, index1, index2, distance=minDist)
                                or overlaps(table_list, index1, index2)):
                            boxes_found_to_be_merged = 1
                            extend_box_and_drop(table_list, index1, index2)

    return table_list


def intersects_in_one_col(box1, box2):
    """
    checks if two regions' horizontal bounds(widths) share an inner bound. (tow regions are on top of each other)

    :param box1: a pandas row series
    :param box2: a pandas row series
    :return: boolean (if two regions horizontal bounds(widths) share an inner bound)
    """
    return max(box1['left'], box2['left']) <= min(box1['X_coordinate2'], box2['X_coordinate2'])


def intersects_in_one_row(box1, box2):
    """
    checks if if two regions' vertical bounds(heights) share an inner bound (are in one row).

    :param box1: a pandas row series
    :param box2: a pandas row series
    :return: boolean (if two regions vertical bounds(heights) share an inner bound)
    """
    return max(box1['top'], box2['top']) <= min(box1['top'] + box1['height'], box2['top'] + box2['height'])


def is_in_one_row(table_list, index1, index2):
    """
    Similar to "intersects_in_one_row" but with list as input

    :param table_list: a complete list of the regions coordinates
    :param index1: index of the first region in the list
    :param index2: index of the second region in the list
    :return:  boolean (if two regions vertical bounds(heights) share an inner bound)
    """
    return max(table_list[index1][0], table_list[index2][0]) < min(table_list[index1][2], table_list[index2][2])


def overlaps(table_list, index1, index2):
    """
    Similar to "intersects_in_one_col" but with list as input
    Is used after being in one row is verified.

    :param table_list: a complete list of the regions coordinates
    :param index1: index of the first region in the list
    :param index2: index of the second region in the list
    :return:  boolean (if two regions horizontal bounds(widths) share an inner bound)
    """
    return max(table_list[index1][1], table_list[index2][1]) <= min(table_list[index1][3], table_list[index2][3])


def are_close(table_list, index1, index2, distance=50):
    """
    Checks if two masks are close to each other by distance(default=50)

    :param table_list: a complete list of the regions coordinates
    :param index1: index of the first region in the list
    :param index2: index of the second region in the list
    :param distance: minumum distance threshold to merge two regions
    :return: boolean (if two regions are close enough)
    """
    return abs(table_list[index1][1] - table_list[index2][3]) < distance


def img_mask_cut(image_array, box):
    """
    cuts array of the mask from the image_array

    :param image_array: 2D image array
    :param box: a pandas row serries which includes coordinates
    :return: image cut array of the region specified with the coordinates in box
    """
    return image_array[box['top']: box['Y_coordinate2'],
                       box['left']: box['X_coordinate2']]


def extend_box_and_drop(table_list, index1, index2):
    """
    In the table_list, which includes all the coordinates of all masks,
    extends the coordinates of the mask with index1 in the list, so that
    its region will cover the mask with index2. Then, it deletes the the
    list element with index2 (since we don't need it anymore and its region
    is now covered by the index1 mask. Thus, the two masks are considered
    merged in the end.

    Structure of table_list : [
                            [top, left, Y_coordinate2, X_coordinate2],
                            [top, left, Y_coordinate2, X_coordinate2],
                            [top, left, Y_coordinate2, X_coordinate2],
                            ...
                              ]



    :param table_list: a complete list of the regions coordinates
    :param index1: index of the first region in the list
    :param index2: index of the second region in the list
    :return: an updated table_list in which the boxes with index1 and index2 are merged
    """
    # orders in table_list : [top, left, Y_coordinate2, X_coordinate2]
    # Here min means extending the top box border more to top to cover the topmost borders of the two boxes.
    table_list[index1][0] = min(table_list[index1][0], table_list[index2][0],
                                table_list[index1][2], table_list[index2][2])
    # Here min means extending the left box border more to left to cover the leftest borders of the two boxes.
    table_list[index1][1] = min(table_list[index1][1], table_list[index2][1],
                                table_list[index1][3], table_list[index2][3])
    # Here max means extending the bottom box border more below to cover the lowest borders of the two boxes.
    table_list[index1][2] = max(table_list[index1][0], table_list[index2][0],
                                table_list[index1][2], table_list[index2][2])
    # Here max means extending the right box border more to right to cover the rightest borders of the two boxes.
    table_list[index1][3] = max(table_list[index1][1], table_list[index2][1],
                                table_list[index1][3], table_list[index2][3])
    del table_list[index2]


# In[73]:


def show_labels(boxes_coord, image_label_overlay, show=1, saving_name=None, save_loc=None):
    """
    Used for demonstrating the found regions (masks) on an image

    :param boxes_coord: pandas DataFrame (output of the sckitting function) which includes the regions coordinates
    :param image_label_overlay: numpy array of the corresponding image
    :param show: whether to plot or not
    :param saving_name: name to be used for saving the masked image
    :param save_loc: path to be used for saving the masked image
    :return:
    """
    fig, ax = plt.subplots(figsize=(50, 50))
    for index, box in boxes_coord.iterrows():
        rect = mpatches.Rectangle((box['left'], box['top']),
                                  box['X_coordinate2'] - box['left'],  # this is the width
                                  box['Y_coordinate2'] - box['top'],  # this is the height
                                  fill=False,
                                  edgecolor='blue',
                                  linewidth=5)
        ax.add_patch(rect)
    plt.tight_layout()
    ax.imshow(image_label_overlay, cmap=plt.cm.copper)
    plt.yticks(np.arange(0, image_label_overlay.shape[0] + 1, 100))
    plt.xticks(np.arange(0, image_label_overlay.shape[1] + 1, 100), rotation=90)
    if saving_name is not None:
        savefig(save_loc + '/' + saving_name)
    if show:
        plt.show()
    else:
        plt.close(fig)


def clean_and_calculate_columns(boxes, min_area=1000, aspect_ratio_bound_1=50, aspect_ratio_bound_2=1 / 5):
    """
    Removes masks in the boxes DataFrame that make no sense since either their area is too small or
    their aspect ratio is not normal and cannot correspond to a box that has a word inside it.

    :param boxes: pandas dataframe (output of the sckitting function) which includes the regions coordinates
    :param min_area: minimum area a region should have not to be deleted
    :param aspect_ratio_bound_1: minimum aspect ratio a region should have not to be deleted
    :param aspect_ratio_bound_2: maximum aspect ratio a region should have not to be deleted
    :return: DataFrame from input without noise regions
    """
    for idx, row in boxes.iterrows():
        width = row['X_coordinate2'] - row['left']
        height = row['Y_coordinate2'] - row['top']
        aspect_ratio = width / height
        area = height * height
        if area < min_area or aspect_ratio > aspect_ratio_bound_1 or aspect_ratio < aspect_ratio_bound_2:
            boxes = boxes.drop(idx, inplace=False)
        else:
            boxes.at[idx, 'Aspect_ratio(width/height)'] = aspect_ratio
            boxes.at[idx, 'Area'] = area
            boxes.at[idx, 'width'] = width
            boxes.at[idx, 'height'] = height
            boxes.at[idx, 'center_x'] = row['left'] + width / 2
            boxes.at[idx, 'center_y'] = row['top'] + height / 2
            boxes.at[idx, 'text'] = 'None'
    boxes['group_id'] = boxes.index
    return boxes


def inner_table_texts(image_array, minDist=40, make_img_label_overlay=1):
    """
    Given an image_array, first finds the table fields and the for each field find the characters inside them and
    merge them together if they are close by MinDist distance or overlap.
    It returns a DataFrame that includes the information such as coordinates of the masks of words only for
    inside the table fields.

    :param image_array: 2D numpy array of the image
    :param minDist: minumum horizontal distance between the regions to be merged with each other
    :param make_img_label_overlay: whether to make a colored image array of the regions
    :return: Pandas DataFrame of the coordinates of the regions inside tables of the image
    """
    outer_table_image_array = np.copy(image_array)
    boxes_coord, image_overlay = scikitting(image_array,
                                            black_region=False,
                                            clearborder=1,
                                            unite_boxes=0,
                                            minDist=minDist,
                                            area_threshold=5000,
                                            make_img_label_overlay=make_img_label_overlay
                                            )
    labels = [
        'top',
        'left',
        'Y_coordinate2',
        'X_coordinate2',
        'row_id',
        'group_id'
    ]

    in_table_boxes = pd.DataFrame(columns=labels)
    for idx, box in boxes_coord.iterrows():
        mask = img_mask_cut(image_array, box)
        mask = errosion_and_extrapolation(image_array=mask, plot=0)
        sub_boxes, _ = scikitting(mask,
                                  area_threshold=25,
                                  thresholding='nieblack',
                                  make_img_label_overlay=0,
                                  clearborder=0,
                                  unite_boxes=1)

        sub_boxes['left'] += box['left']
        sub_boxes['top'] += box['top']
        sub_boxes['X_coordinate2'] += box['left']
        sub_boxes['Y_coordinate2'] += box['top']
        sub_boxes['row_id'] += 0.5  # to distinguish inner and outer table texts

        outer_table_image_array[box['top'] - 15: box['Y_coordinate2'] + 15,
        box['left'] - 15: box['X_coordinate2'] + 15] = 255

        in_table_boxes = pd.concat([in_table_boxes, sub_boxes], ignore_index=True, sort=False)
    return in_table_boxes, outer_table_image_array, image_overlay


def outer_table_texts(image_array, area_threshold=400):
    """
    Given an image_array ( e.g. an image array that does not have any table inside it), it finds the
    Characters inside the image, merges the together and returns a DataFrame for the corresponding
    information such as masks coordinates.
    similar output to inner_table_texts, but for outside of the tables.

    :param image_array: 2D numpy array of the image
    :param area_threshold: Minimum area of regions to be accepted as mask
    :return: Pandas DataFrame of the coordinates of the regions outside tables of the image
    """
    boxes, _ = scikitting(image_array,
                          area_threshold=area_threshold,
                          make_img_label_overlay=0,
                          unite_boxes=1,
                          black_region=True,
                          thresholding='niblack',
                          clearborder=1)
    return boxes


def Auto_mask_image(img_path, saving_path=None):
    """
    Given an image path, it finds characters, merges them together and return a DataFrame that
    includes the information such as the coordinates of masks in the image.

    step1: extracts the masks only for inside table fields. and remove table fields from the image array for step2.
    step2: takes the image array without table fields and extracts masks of the words inside it.
    step3: concatanation of the DataFrame results from step1 and step2.
    step4: cleaning the image from masks whose area or aspect ratio do not make sense to be mask of a word
    step5: if a saving_path: saves the DataFrame as an xlsx file.

    :param img_path: path(including the name and its format) of the target image
    :param saving_path: path for saving an xlsx file which includes coordinates of the regions in image that includes words/numbers
    :return: Pandas DataFrame version of the above xlsx file
    """
    IMG = cv2.imread(img_path, 0)
    
    iner_boxes, outer_img_array, _ = inner_table_texts(image_array=IMG, minDist=40, make_img_label_overlay=0)
    
    outer_img_array_eroded = errosion_and_extrapolation(image_array=outer_img_array, plot=0)
    
    outer_boxes = outer_table_texts(outer_img_array_eroded, area_threshold=1)
    
    all_boxes = pd.concat([outer_boxes, iner_boxes], ignore_index=True, sort=False)
    all_boxes = clean_and_calculate_columns(all_boxes, min_area=700, aspect_ratio_bound_1=100,
                                            aspect_ratio_bound_2=1 / 5)
    if saving_path is not None:
        all_boxes.to_excel(saving_path + '/image_mask_coordinates.xlsx', engine='xlsxwriter')
    return all_boxes


def read_write_txts(IMG, boxes):
    """
    given an image and masks coordinates, reads all texts inside those masks

    :param IMG: 2D numpy array of the target image
    :param boxes: pandas DataFrame which includes the coordinates of masks
    :return: same DataFrame as input but with the text fields filled
    """
    for idx, row in boxes.iterrows():
        mask = img_mask_cut(IMG, row)
        mask = errosion_and_extrapolation(image_array=mask, plot=0)
        mask = np.pad(mask, 10, 'constant', constant_values=255)
        text = pytesseract.image_to_string(mask)
        boxes.at[idx, 'text'] = text
    return boxes


def get_upper_labels(boxes, target_box, until_n_boxes=3, n_letters_thr=3):
    """
    Given boxes (DataFrame of all masks' information), and a target_box, it returns
    a sorted list of possible labels of that target box that are above it and have at least
    3 letters inside them.

    Among the sorted boxes that also pass the criteria of numbers of letters inside
    them until_n_boxes of them is returned.

    :param boxes: Pandas DataFrame with coordinates and text fields of each mask
    :param target_box: Pandas row serries of the target mask for which we want to know the upper labels
    :param until_n_boxes: number of regions above the target region to be considered as label
    :param n_letters_thr: Threshold used for neglecting regions filled with number
    :return:
    """
    upper_boxes = pd.DataFrame()
    for idx, row in boxes.iterrows():
        if intersects_in_one_col(box1=row, box2=target_box):
            num_of_letters = sum(c.isalpha() for c in row.text)
            if num_of_letters >= n_letters_thr:
                upper_boxes = pd.concat([upper_boxes, row.to_frame().T])
    upper_boxes = upper_boxes.query('top < ' + str(target_box.top)).sort_values(by=['top'], ascending=False)
    return upper_boxes.text[:until_n_boxes].tolist()


def get_left_labels(boxes, target_box, until_n_boxes=3, n_letters_thr=3):
    """
    Given boxes (DataFrame of all masks' information), and a target_box, it returns
    a sorted(based on left coordinates) list of possible labels of that target box
    that are left side of it and have at least 3 letters inside them.

    Among the sorted boxes that also pass the criteria of numbers of letters inside
    them until_n_boxes of them is returned.

    :param boxes: Pandas DataFrame with coordinates and text fields of each mask
    :param target_box: Pandas row serries of the target mask for which we want to know the left labels
    :param until_n_boxes: number of regions above the target region to be considered as label
    :param n_letters_thr: Threshold used for neglecting regions filled with number
    :return:
    """
    left_boxes = pd.DataFrame()
    for idx, row in boxes.iterrows():
        if intersects_in_one_row(box1=row, box2=target_box):
            num_of_letters = sum(c.isalpha() for c in row.text)
            if num_of_letters >= n_letters_thr:
                left_boxes = pd.concat([left_boxes, row.to_frame().T])
    left_boxes = left_boxes.query('left < ' + str(target_box.left)).sort_values(by=['left'], ascending=False)
    return left_boxes.text[:until_n_boxes].tolist()





