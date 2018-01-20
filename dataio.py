import os, shutil, random, time, json
from glob import glob

def maybe_train_test_split(dataset_dir, split_ratio=0.2):
    """
    Prepares directories with training and test pictures. Creates two new dirs 'training' and 'test' and moves randomly
    files to them, unless those two dirs already exist, then does nothing. Keeps dataset_dir intact
    :param dataset_dir: dataset directory where 'training' and 'test' dirs will be created
    :param split_ratio: ratio of test samples from range [0, 1]
    :return: (train path, test path) pair
    """
    datadir_prefix, dirname = os.path.split(dataset_dir)
    split_datadir = os.path.join(datadir_prefix, dirname + '_split')
    train_path = os.path.join(split_datadir, 'train')
    test_path = os.path.join(split_datadir, 'test')

    if not os.path.exists(split_datadir):
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        for class_name, imgs in scan_dataset(dataset_dir).items():
            random.shuffle(imgs)
            split_index = int(len(imgs) * (1-split_ratio))
            train_images, test_images = imgs[:split_index], imgs[split_index:]

            copy_files(train_images, os.path.join(train_path, class_name))
            copy_files(test_images, os.path.join(test_path, class_name))

    return train_path, test_path

def copy_files(files, directory):
    """Creates new directory and copy files to it"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Created directory: {}".format(directory))

    for f in files:
        shutil.copy(f, directory)
    print("Copied {} files.\n".format(len(files)))


def scan_dataset(dataset_dir):

    sub_dir = map(lambda d: os.path.basename(d.rstrip("/")), glob(os.path.join(dataset_dir, '*/')))

    data_dic = {}
    for class_name in sub_dir:
        imgs = glob(os.path.join(dataset_dir, class_name, "*.jpg"))

        data_dic[class_name] = imgs

    return data_dic

def mk_unique_filename(prefix):
    timestamp = time.strftime('_%Y_%m_%d_%H_%M_%S', time.localtime())
    return prefix + timestamp

def history_to_json(history, run_time=None, filename=None):

    data = {
        'history': history,
        'run_time': run_time
    }

    if filename is None:
        filename = mk_unique_filename('training') + '.json'

    with open(filename, 'w') as jsonfp:
        json.dump(data, jsonfp)

    return filename