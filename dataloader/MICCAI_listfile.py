import os
import os.path

IMG_EXTENSIONS = ['.PNG', '.png']
DISP_EXTENSIONS = ['.PFM', '.pfm']


def is_img_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_disp_file(filename):
    return any(filename.endswith(extension) for extension in DISP_EXTENSIONS)

def dataloader(filepath):

    train_path = filepath + '/train'
    val_path = filepath + '/val'
    test_path = filepath + '/test'

    all_left_img = []
    all_right_img = []
    all_left_disp = []
    val_left_img = []
    val_right_img = []
    val_left_disp = []
    test_left_img = []
    test_right_img = []

    # this part get all the datasets path (dataset1, dataset2, dataset3)
    train_sets = [os.path.join(train_path,s) for s in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, s))]
    val_sets = [os.path.join(val_path,s) for s in os.listdir(val_path) if os.path.isdir(os.path.join(val_path, s))]
    test_sets = [os.path.join(test_path,s) for s in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, s))]

    # this part get all the keyframe path in one single set
    train_keyframe = []
    val_keyframe = []
    test_keyframe = []
    for ts in train_sets:
        sub_keyframe = os.listdir(ts)
        for skf in sub_keyframe:
            if skf.find('ignore'):
                train_keyframe.append(os.path.join(ts, skf))

    for vs in val_sets:
        sub_keyframe = os.listdir(vs)
        for skf in sub_keyframe:
            if skf.find('ignore'):
                val_keyframe.append(os.path.join(vs, skf))

    for test_sub in test_sets:
        sub_keyframe = os.listdir(test_sub)
        for skf in sub_keyframe:
            if skf.find('ignore'):
                test_keyframe.append(os.path.join(test_sub, skf))



    # get img file
    for tk in train_keyframe:
        # get all left_img
        left_img_path = tk + '/left_finalpass/'
        right_img_path = tk  + '/right_finalpass/'
        left_disp_path = tk  + '/transformed_disp_pfm/'
        for disp in os.listdir(left_disp_path):
            if is_disp_file(left_disp_path + '/' + disp):
                all_left_disp.append(left_disp_path + disp)
                all_left_img.append(left_img_path + disp.split('.')[0] + '.png')
                all_right_img.append(right_img_path + disp.split('.')[0] + '.png')

    for vk in val_keyframe:
        # get all left_img
        left_img_path = vk + '/left_finalpass/'
        right_img_path = vk  + '/right_finalpass/'
        left_disp_path = vk  + '/transformed_disp_pfm/'
        for disp in os.listdir(left_disp_path):
            if is_disp_file(left_disp_path + '/' + disp):
                val_left_disp.append(left_disp_path + disp)
                val_left_img.append(left_img_path + disp.split('.')[0] + '.png')
                val_right_img.append(right_img_path + disp.split('.')[0] + '.png')


    for test_k in test_keyframe:
        # get all left_img
        left_img_path = test_k + '/left_finalpass/'
        right_img_path = test_k  + '/right_finalpass/'
        for disp in os.listdir(left_img_path):
            test_left_img.append(left_img_path + disp.split('.')[0] + '.png')
            test_right_img.append(right_img_path + disp.split('.')[0] + '.png')

    all_left_img = all_left_img + val_left_img
    all_right_img = all_right_img + val_right_img
    all_left_disp = all_left_disp + val_left_disp

    return all_left_img, all_right_img, all_left_disp, val_left_img, val_right_img, val_left_disp, test_left_img, test_right_img



# def test_loader():
#     path = '/media/xiran_zhang/TOSHIBA EXT/MICCAI'
#     dataloader(path)
#
#
# if __name__  == '__main__':
#     test_loader()

