# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
from PIL import Image, ImageChops, ImageOps, ImageEnhance
from skimage import transform
from skimage import io
from multiprocessing import Pool
from functools import partial
import numpy as np
import argparse
import os
import random
import math


def get_image_paths(folder):
    file_list = []
    dir_list = []
    for root, subdirs, files in os.walk(folder, topdown=False):
        rel_path = os.path.relpath(root, folder)
        jpeg_files = [os.path.join(rel_path, f) for f in files if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg')]
        if rel_path and (jpeg_files or subdirs):
            dir_list.append(rel_path)
        if jpeg_files:
            file_list.append(jpeg_files)
    return file_list, dir_list[::-1]


def rotated_rect_with_max_area(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0,0

    angle = math.radians(angle)
    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr, hr = (x/sin_a, x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr, hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

    return wr, hr


def scale_image(image, size=(299,299), keep_aspect=False, pad=False):
    if keep_aspect:
        image.thumbnail(size, Image.ANTIALIAS)
        image_size = image.size
        if pad:
            thumb = image.crop((0, 0, size[0], size[1]))
            offset_x = max((size[0] - image_size[0]) // 2, 0)
            offset_y = max((size[1] - image_size[1]) // 2, 0)
            thumb = ImageChops.offset(thumb, offset_x, offset_y)
        else:
            thumb = ImageOps.fit(image, size, Image.ANTIALIAS, 0.0, (0.5, 0.5))
    else:
        thumb = image.resize(size, Image.ANTIALIAS)

    return thumb


def distort_image_pil(image, size=(299, 299), keep_aspect=False, pad=False):

    rot = random.randint(-10, 10)
    
    image_size = image.size
    distorted_image = image.rotate(rot, resample=Image.BICUBIC, expand=False)
    wr, hr = rotated_rect_with_max_area(image_size[0], image_size[1], rot)
    wr = int(wr)
    hr = int(hr)
    crop_rect = (
        (image_size[0]-wr) // 2,
        (image_size[1]-hr) // 2,
        (image_size[0]+wr) // 2,
        (image_size[1]+hr) // 2) 
    if wr > size[0] * 2:
        max_shift = (wr - size[0] * 2)
        odd = int(max_shift) % 2
        max_shift //= 2
        offset = random.randint(-max_shift, max_shift)
        shift = (max_shift + offset, max_shift - offset + odd)
    else:
        shift = (0, 0)
    crop_rect = (crop_rect[0] + shift[0], crop_rect[1], crop_rect[2] - shift[1], crop_rect[3])
    distorted_image = distorted_image.crop(crop_rect)

    distorted_image = scale_image(distorted_image, size, keep_aspect, pad)

    #brightness = ImageEnhance.Brightness(thumb)
    #thumb = brightness.enhance(random.uniform(.75, 1.25))
    #contrast = ImageEnhance.Contrast(thumb)
    #thumb = contrast.enhance(random.uniform(.75, 1.25))

    return distorted_image


def distort_image_sk(image, size=(299, 299)):

    rot = np.deg2rad(random.randint(-10, 10))
    sheer = np.deg2rad(random.randint(-10, 10))

    shape = image.shape
    shape_size = shape[:2]
    center = np.float32(shape_size) / 2. - 0.5

    pre = transform.SimilarityTransform(translation=-center)
    affine = transform.AffineTransform(rotation=rot, shear=sheer, translation=center)
    tform = pre + affine

    distorted_image = transform.warp(image, tform.params, mode='reflect')

    distorted_image = transform.resize(distorted_image, size)

    return distorted_image


def calc_stats(image_paths, source_dir):

    image_count = 0
    image_var = []
    image_mean = []
    pixels_per_image = 299 * 299
    for path in image_paths:
        try:
            abs_path = os.path.join(source_dir, path)
            image = Image.open(abs_path)
        except:
            continue
        image_array = np.asarray(image)
        mean = np.mean(image_array, axis=(0, 1))
        if (mean[0] < 16):
            print("Invalid file :s" % path)
            continue
        image_mean.append(mean)
        var = np.var(image_array, axis=(0, 1))
        image_var.append(var)
        image_count += 1

    stats = {'count': image_count, 'mean': np.mean(image_mean, axis=0), 'var': np.mean(image_var, axis=0)}
    return stats


def safe_mkdir(path):
    try:
        os.makedirs(path)
    except OSError as err:
        if not os.path.isdir(path):
            print("Error %s making directory" % err)
            raise


def gen_filename(root, file_name, tag=''):
    file_root, file_ext = os.path.splitext(file_name)
    if not file_ext:
        file_ext = '.jpg'
    if tag:
        file_name = file_root + '_' + tag + file_ext
    else:
        file_name = file_root + file_ext
    return os.path.join(root, file_name)


def process_multiple(image_paths, source_dir, dest_dir, distortion_count=0):
    for file in image_paths:
        abs_image_path = os.path.join(source_dir, file)
        dest_path = os.path.join(dest_dir, file)
        process_file(abs_image_path, dest_path, distortion_count=distortion_count)


def process_file(source_path, dest_path, distortion_count=0):
    image = Image.open(source_path)

    out_dir, out_file = os.path.split(dest_path)
    if not out_file:
        print('Invalid destination file')
        return
    
    #scaled = scale_image(image, keep_aspect=False)
    #scaled.save(gen_filename(out_dir, out_file))
    for count in range(distortion_count):
        distorted = distort_image_pil(image, keep_aspect=False)
        distorted.save(gen_filename(out_dir, out_file, 'a-%d' % (count + 1)))

        distorted2 = distort_image_sk(np.asarray(image))
        io.imsave(gen_filename(out_dir, out_file, 'b-%d' % (count + 1)), distorted2)


def combine_stats(stats):
    image_count_total = 0
    image_mean = np.float64(0)
    image_var = np.float64(0)
    for stat_item in stats:
        image_count = stat_item['count']
        image_count_total += image_count
        image_mean += stat_item['mean'] * image_count
        image_var += stat_item['var'] * (image_count + 1)
    image_var /= (image_count_total - len(stats))
    image_mean /= image_count_total
    image_std = np.sqrt(image_var)

    return image_mean, image_std


def process_dir(source_dir, dest_dir, distortion_count=0, validation_percent=0):
    image_files, image_dirs = get_image_paths(source_dir)
    if not image_files:
        return

    validation_files = []
    if validation_percent > 0:
        random.shuffle(image_files)
        part = math.ceil((validation_percent / 100.) * len(image_files))
        validation_files = image_files[:part]
        image_files = image_files[part:]

    for rel_dir in image_dirs:
        directory = os.path.join(dest_dir, rel_dir)
        safe_mkdir(directory)
        if validation_percent > 0:
            directory = os.path.join(dest_dir, 'val/', rel_dir)
            safe_mkdir(directory)

    pool = Pool(4)
    stats = pool.map(
        partial(process_multiple, source_dir=source_dir, dest_dir=dest_dir, distortion_count=distortion_count),
        image_files)
    image_mean, image_std = combine_stats(stats)
    print(image_mean, image_mean/255, image_std, image_std/255)
    if validation_percent > 0:
        val_dir = os.path.join(dest_dir, 'val/')
        pool.map(
            partial(process_multiple, source_dir=source_dir, dest_dir=val_dir, distortion_count=0),
            validation_files)
    pool.close() 
    pool.join()


def directory_stats(source_dir):
    image_files, _ = get_image_paths(source_dir)
    if not image_files:
        return

    pool = Pool(4)
    stats = pool.map(partial(calc_stats, source_dir=source_dir), image_files)
    pool.close()
    pool.join()
    image_mean, image_std = combine_stats(stats)
    print(image_mean, image_mean/255, image_std, image_std/255)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('dest')
    args = parser.parse_args()
    source_path = args.source
    dest_path = args.dest

    if (os.path.isdir(source_path)):
        #process_dir(source_path, dest_path, 5, True)
        directory_stats(source_path)
    elif (os.path.isfile(source_path)):
        process_file(source_path, dest_path, 2)
    else:
        print("%s is not a valid file or folder" % source_path)
    
if __name__ == "__main__":
    main()

