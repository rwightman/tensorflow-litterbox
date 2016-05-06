import pandas as pd
import argparse
import os
import shutil
import random

DRIVER_FILE = '../data/driver_imgs_list.csv'

DEL_DRIVERS = ['p081']
VAL_DRIVERS = {'p072': .8, 'p047': 1.0, 'p056': 0.4, 'p022': 0.4, 'p015': 0.4}


def safe_mkdir(path):
    try:
        os.makedirs(path)
    except OSError as err:
        if not os.path.isdir(path):
            print("Error %s making directory" % err)
            raise


def move(drivers, source_dir, dest_basedir, dest_subdir):
    for cat, img in zip(drivers.classname, drivers.img):
        dest_dir = os.path.join(dest_basedir, dest_subdir)
        source_file = os.path.join(source_dir, cat, img)
        dest_file = os.path.join(dest_dir, cat, img)
        dest_root = os.path.dirname(dest_file)
        if not os.path.exists(dest_root):
            print("Creating dest folder %s" % dest_root)
            safe_mkdir(dest_root)
        print('moving %s to %s' % (source_file, dest_file))
        try:
            shutil.move(source_file, dest_file)
        except (IOError, os.error) as why:
            print("IO Error %s" % why)
        except shutil.Error as err:
            print("Error (%s) moving file" % err)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('dest')
    args = parser.parse_args()
    source_path = args.source
    dest_path = args.dest

    if not os.path.isdir(source_path):
        print("%s is not a valid folder" % source_path)
        exit(-1)

    drivers = pd.read_csv(DRIVER_FILE, header=0, index_col=False)

    driver_count = pd.value_counts(drivers['subject'].values)

    del_drivers = drivers[drivers.subject.isin(DEL_DRIVERS)]

    filtered = []
    for val_driver, percent in VAL_DRIVERS.items():
        sel = drivers[drivers.subject == val_driver]
        if percent < 1.0:
            sel = sel.sample(frac=percent)
        print(len(sel))
        filtered.append(sel)
    val_drivers = pd.concat(filtered)

    move(val_drivers, source_dir=source_path, dest_basedir=dest_path, dest_subdir='val')
    move(del_drivers, source_dir=source_path, dest_basedir=dest_path, dest_subdir='del')

if __name__ == "__main__":
    main()

