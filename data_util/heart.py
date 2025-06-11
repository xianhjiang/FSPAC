import os
import json
import copy
import re

import numpy as np
import utils
import nibabel as nib
from skimage import exposure
from skimage import transform
from .liver import FileManager
from .liver import Dataset as BaseDataset

class ACDCDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.patient_folders = [folder for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder))]

    def load_nifti(self, folder, filename):
        # print(f"路径是 ：{self.root_dir,'----',folder,'----',filename}")
        file_path = os.path.join(self.root_dir, folder, filename)
        # print(f"file_path路径是 ：{file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No such file or no access: '{file_path}'")

        return nib.load(file_path).get_fdata()

    def load_acdc_data(self, folder):
        # print(f"folder是---{folder}")
        data = {}
        data['id'] = folder  # 添加一个表示患者ID的键
        match = re.search(r'patient\d{3}', folder)  # 添加一个表示患者ID的键
        if match :
            data['id'] = match.group()

        # Load 4D image
        data['4d_image'] = self.load_nifti(data['id'], f"{data['id']}_4d.nii.gz")

        # Load frame 01 image and its corresponding label
        data['frame01_image'] = self.load_nifti(data['id'], f"{data['id']}_frame01.nii.gz")
        data['frame01_label'] = self.load_nifti(data['id'], f"{data['id']}_frame01_gt.nii.gz")

        # 定义匹配模式
        pattern = re.compile(r'_frame((?!01)\d{2})\.nii\.gz')  # 匹配形如 *_frameXX.nii.gz，其中 XX 不为 "01"
        file_path = os.path.join(self.root_dir, data['id'])
        file_list = os.listdir(file_path)
        # 过滤出符合条件的文件名 就只会匹配一个出来
        matching_files = [filename for filename in file_list if pattern.search(filename)]
        # 加载符合条件的文件
        for filename in matching_files:
            frame_number = pattern.search(filename).group(1)  # 提取帧数
            data['frame02_image'] = self.load_nifti(data['id'], filename)
            label_filename = f"{data['id']}_frame{frame_number}_gt.nii.gz"  # 构造标签文件名
            data['frame02_label'] = self.load_nifti(data['id'], label_filename)


        return data

# 数据集类的定义
class Dataset(BaseDataset):
    def __init__(self, split_path, paired=False, task=None, batch_size=None, mode=1):
        with open(split_path, 'r') as f:
            config = json.load(f)

        self.subset = {}
        # print(f"self.files--{self.files}")
        for k, v in config['subsets'].items():
            # 新增ACDCDataset实例
            if mode == 0:
                if "train" in k:
                    self.acdc_dataset = ACDCDataset(config["files"]["acdcTraining"]["path"])
                elif "test" in k:
                    self.acdc_dataset = ACDCDataset(config["files"]["acdcTesting"]["path"])
                else:
                    continue
            elif "train" in k and mode == 1:
                self.acdc_dataset = ACDCDataset(config["files"]["acdcTraining"]["path"])
            elif "test" in k and mode == 2:
                self.acdc_dataset = ACDCDataset(config["files"]["acdcTesting"]["path"])
            elif "calculateDice" in k and mode == 3:
                self.acdc_dataset = ACDCDataset(config["files"]["calculateDice"]["path"])
            else:
                continue
            self.subset[k] = {}
            # print(f"k,v{k,v}")
            # i = 0
            for entry in v:
                # 这里可以进行处理acdc数据
                # self.subset[k][entry] = self.files[entry]
                self.subset[k][entry] = self.acdc_dataset.load_acdc_data(entry)
                # print(f"self.files[entry]---{self.files[entry]}")
                # i = i + 1
                # if i == 10:
                #     break


        self.paired = paired

        def convert_int(key):
            try:
                return int(key)
            except ValueError as e:
                return key
        self.schemes = dict([(convert_int(k), v)
                             for k, v in config['schemes'].items()])

        for k, v in self.subset.items():
            print('Number of data in {} is {}'.format(k, len(v)))

        self.task = task
        if self.task is None:
            self.task = config.get("task", "registration")
        if not isinstance(self.task, list):
            self.task = [self.task]

        # print(f"self.task---{self.task}")

        self.image_size = config.get("image_size", [128, 128, 128])  # 调整以匹配ACDC数据集
        self.segmentation_class_value = config.get(
            'segmentation_class_value', None)

        if 'atlas' in config:
            self.acdc_dataset = ACDCDataset(config["files"]["acdcTraining"]["path"])
            self.atlas = self.acdc_dataset.load_acdc_data(config['atlas'])
        else:
            self.atlas = None

        if paired:
            self.atlas = None

        self.batch_size = batch_size
        self.fixed_seg = None



    def preprocess_image(self, image):
        # 示例预处理：直方图均衡化和调整大小
        # image = exposure.equalize_hist(image)
        image = transform.resize(image, self.image_size, anti_aliasing=True)

        return image

    def center_crop(self, volume):
        slices = [slice((os - ts) // 2, (os - ts) // 2 + ts) if ts < os else slice(None, None)
                  for ts, os in zip(self.image_size, volume.shape)]
        volume = volume[slices]

        ret = np.zeros(self.image_size, dtype=volume.dtype)
        slices = [slice((ts - os) // 2, (ts - os) // 2 + os) if ts > os else slice(None, None)
                  for ts, os in zip(self.image_size, volume.shape)]
        ret[slices] = volume

        return ret

    @staticmethod
    def generate_atlas(atlas, sets, loop=False):
        sets = copy.copy(sets)
        while True:
            if loop:
                np.random.shuffle(sets)
            for d in sets:
                yield atlas, d
            if not loop:
                break

    @staticmethod
    def generate_mypairs(sets, loop=False):
        sets = copy.copy(sets)
        while True:
            if loop:
                np.random.shuffle(sets)
            for d in sets:
                yield d
            if not loop:
                break

    def generator(self, subset, batch_size=None, loop=False, aug=False):
        # print("Generating data for subset:", subset)
        if batch_size is None:
            batch_size = self.batch_size
        scheme = self.schemes[subset]
        if 'registration' in self.task:
            if self.atlas is not None:
                # print(f"self.atlas is not None---{self.atlas}")
                generators, fractions = zip(*[(self.generate_atlas(self.atlas, list(
                    self.subset[k].values()), loop), fraction) for k, fraction in scheme.items()])
                print(f"generators---{generators}")
            else:
                # print(f"self.atlas is None, 做每个病人不同时期的配准")
                # generators, fractions = zip(
                #     *[(self.generate_pairs(list(self.subset[k].values()), loop), fraction) for k, fraction in scheme.items()])

                generators, fractions = zip(
                    *[(self.generate_mypairs(list(self.subset[k].values()), loop), fraction) for k, fraction in scheme.items()])


            while True:
                i = 0
                flag = True
                ret = dict()
                for gen in generators:
                    try:
                        while True:
                            # 使用ACDCDataset加载数据
                            d1 = next(gen)
                            # print("Data structure for d1:", d1)
                            # print("Data structure for d2:", d2)
                            break
                    except StopIteration:
                        flag = False
                        break


                    # 假设体积数据存储在'4d_image'，'frame01_image'等中。
                    # print(f"d1_acdc--{d1}")
                    # print(f"d2_acdc--{d2}")
                    fixed = np.array(d1['frame01_image'])
                    moving = np.array(d1['frame02_image'])

                    if aug:
                        moving = utils.perturb(moving, tmp_level=5)

                    # 预处理图像
                    fixed = self.preprocess_image(fixed)
                    moving = self.preprocess_image(moving)

                    # ret['fixed'] = utils.tensor(fixed/255.)[None, None, ..., 0]
                    ret['fixed'] = utils.tensor(fixed/255.)[None, None, ...]
                    ret['moving'] = utils.tensor(moving/255.)[None, None, ...]

                    # print("Size of fixed tensor:", ret['fixed'].size())
                    # print("Size of moving tensor:", ret['moving'].size())

                    if 'frame01_label' in d1:
                        # print("frame01_label进来了")
                        # ret['fixed_seg'] = np.array(d1['frame01_label'])[None, None, ...]
                        fixed_seg = self.preprocess_image(np.array(d1['frame01_label']))
                        # fixed_seg = utils.tensor(fixed_seg/255.)[None, None, ...]
                        fixed_seg = utils.tensor(fixed_seg)[None, None, ...]
                        ret['fixed_seg'] = fixed_seg

                    if 'frame02_label' in d1:
                        # ret['moving_seg'] = np.array(d2['frame01_label'])[None,None,...]
                        moving_seg = self.preprocess_image(np.array(d1['frame02_label']))
                        # moving_seg = utils.tensor(moving_seg/255.)[None, None, ...]
                        moving_seg = utils.tensor(moving_seg)[None, None, ...]
                        ret['moving_seg'] = moving_seg
                    i += 1

                if flag:
                    yield ret
                else:
                    break
