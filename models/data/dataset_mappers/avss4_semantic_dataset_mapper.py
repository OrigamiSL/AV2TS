import copy
import logging
import pickle

import numpy as np
import torch
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances
from torch.nn import functional as F


class AVSS4_SemanticDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by AV2TS for Audio-Visual Segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """
    
    # @configurable difined by detron2, the arguement before  * can be extracted from 'cfg', 
    # the arguement after * can be defined specifically
    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        pre_sam,
    ):
        """
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
            pre_sam: whether to use pre mask
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        # self.__class__.__name__ : get the name of the class
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")
        # * Whether to load pre mask
        self.pre_sam = pre_sam

    def load_audio_lm(self, audio_lm_path):
        """Load audio log mel spectrogram from pickle file"""
        with open(audio_lm_path, "rb") as fr:
            audio_log_mel = pickle.load(fr)
        audio_log_mel = audio_log_mel.detach()  # [5, 1, 96, 64]
        return audio_log_mel

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        if is_train:
            if cfg.INPUT.AUGMENTATION == True:
                """
                Resize the image while keeping the aspect ratio unchanged.
                It attempts to scale the shorter edge to the given `short_edge_length`,
                as long as the longer edge does not exceed `max_size`.
                If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.

                short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
                """

                # size = np.random.choice(self.short_edge_length)
                augs = [
                    # If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
                    # resize(min,xxxx)  or (xxxx,max) in the oringin image aspect ratio
                    T.ResizeShortestEdge(
                        cfg.INPUT.MIN_SIZE_TRAIN, #train: [int(x * 0.1 * 224) for x in range(5, 21)], test:224
                        cfg.INPUT.MAX_SIZE_TRAIN, #896
                        cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,# 'choice'
                    )
                ]
                
                # random crop (224, 224)
                if cfg.INPUT.CROP.ENABLED:
                        # ENABLED: True
                        # TYPE: "absolute"
                        # SIZE: (224, 224)
                        # SINGLE_CATEGORY_MAX_AREA: 1.0
                        # SEM_SEG_HEAD.IGNORE_VALUE: 255
                        augs.append(
                        T.RandomCrop_CategoryAreaConstraint(
                            cfg.INPUT.CROP.TYPE,
                            cfg.INPUT.CROP.SIZE,
                            cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                            cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                        )
                    )
                # A color related data augmentation used in Single Shot Multibox Detector (SSD).
                if cfg.INPUT.COLOR_AUG_SSD:
                    augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))

                # 50% horizontally flip
                augs.append(T.RandomFlip())
            else:
                augs = []
        else:

            augs = []  # No augmentation for validation and test

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        # num_classes, ignore_label: deprecated argument
        ignore_label = meta.ignore_label

        # Whether to use pre sam
        
        pre_sam = cfg.MODEL.PRE_SAM.USE_PRE_SAM # False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT, #'RGB‘
            "ignore_label": ignore_label, # deprecated
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY, # 224  # used in dataset mapper, close it
            "pre_sam": pre_sam, # False
        }
        return ret
        # ret is output for apply_transform_gens

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        images = []

        for image_path in dataset_dict["file_names"]:
            image = utils.read_image(image_path, format=self.img_format)
            images.append(image)
        utils.check_image_size(dataset_dict, images[0])

        if "sem_seg_file_names" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_file_name_list = dataset_dict.pop("sem_seg_file_names")
            only_first_mask = True if len(sem_seg_file_name_list) == 1 else False
            sem_seg_gts = []
            for sem_seg_file_name in sem_seg_file_name_list:
                sem_seg_gt = utils.read_image(sem_seg_file_name)
                # normalize
                sem_seg_gt = sem_seg_gt // 255
                sem_seg_gts.append(sem_seg_gt)
        else:
            sem_seg_gts = None
            raise ValueError("Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(dataset_dict["file_names"]))

        if ("pre_mask_file_names" in dataset_dict) and self.pre_sam:
            pre_mask_file_name_list = dataset_dict.pop("pre_mask_file_names")
            pre_mask_gts = []
            for pre_mask_file_name in pre_mask_file_name_list:
                pre_mask_gt = utils.read_image(pre_mask_file_name, format=self.img_format)
                pre_mask_gts.append(pre_mask_gt)
        else:
            pre_mask_gts = None

        for num_img, image in enumerate(images):
            if num_img == 0:
                # * first image with random augmentation
                aug_input = T.AugInput(image, sem_seg=sem_seg_gts[num_img])
                aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
                image = aug_input.image
                sem_seg_gt = aug_input.sem_seg

            else:
                # * other images with the same augmentation
                image = transforms.apply_image(image)
                if only_first_mask == False:
                    sem_seg_gt = transforms.apply_segmentation(sem_seg_gts[num_img])
                # Pad image and segmentation label here!
                    
            if self.pre_sam: # False
                pre_mask_gt = transforms.apply_image(pre_mask_gts[num_img])
                pre_mask_gt = torch.as_tensor(np.ascontiguousarray(pre_mask_gt.transpose(2, 0, 1)))

            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))) # C,H,W

            if (num_img == 0) or (only_first_mask == False):
                sem_seg_gt = torch.as_tensor(sem_seg_gt.copy())

            if self.size_divisibility > 0:
                image_size = (image.shape[-2], image.shape[-1])
                padding_size = [
                    0,
                    self.size_divisibility - image_size[1],
                    0,
                    self.size_divisibility - image_size[0],
                ]
                image = F.pad(image, padding_size, value=128).contiguous()
                if self.pre_sam:
                    pre_mask_gt = F.pad(pre_mask_gt, padding_size, value=128).contiguous()
                if (num_img == 0) or (only_first_mask == False):
                    sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

            images[num_img] = image
            if (num_img == 0) or (only_first_mask == False):
                sem_seg_gts[num_img] = sem_seg_gt

            if self.pre_sam:
                pre_mask_gts[num_img] = pre_mask_gt

        image_shape = (images[0].shape[-2], images[0].shape[-1])  # h, w

        imgs_tensor = torch.stack(images, dim=0) # N* C* H* W
        masks_tensor = torch.stack(sem_seg_gts, dim=0).unsqueeze(dim=1) #[5,1,224,224] val or [1,1,224,224] train

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.

        dataset_dict["images"] = imgs_tensor

        dataset_dict["sem_segs"] = masks_tensor.float()

        if self.pre_sam:
            pre_masks_tensor = torch.stack(pre_mask_gts, dim=0)  # [5,3,224,224]
            dataset_dict["pre_masks"] = pre_masks_tensor
        # Prepare per-category binary masks
        instances_list = []
        for sem_seg_gt in sem_seg_gts:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                # get the various classes gt
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks]))
                # class_number * H * W
                instances.gt_masks = masks.tensor
            instances_list.append(instances)
        dataset_dict["instances"] = instances_list

        # Prepare audio input
        if "audio_file_name" in dataset_dict:
            audio_file_name = dataset_dict.pop("audio_file_name")
            audio_log_mel = self.load_audio_lm(audio_file_name)

        else:
            raise ValueError("Cannot find 'audio_file_name' for semantic segmentation dataset {}.".format(dataset_dict["file_names"]))
        dataset_dict["audio_log_mel"] = audio_log_mel
        return dataset_dict
