import os


def write_txt(name, msg):
    path_name = 'D:\\deep_sort_yolov3-master\\points_file\\'
    full_path_name = path_name + name + '.txt'
    if not os.path.exists(full_path_name):
        file = open(full_path_name, 'w')
    file = open(full_path_name, 'a')
    file.write(msg)

