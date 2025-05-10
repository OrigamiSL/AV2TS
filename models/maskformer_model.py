from typing import Tuple

import torch
import random
from torch import nn
from torch.nn import functional as F
from detectron2.layers import FrozenBatchNorm2d
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.criterion_ss import SetCriterion_SS
from .modeling.matcher import HungarianMatcher

# audio backbone
from .modeling.audio_backbone.torchvggish import vggish
from .modeling.audio_backbone.patch_backbone.bulid_patch_backbone import Patch_Audio, Patch_Audio_2_dimensional
from .modeling.audio_backbone.CED.build_CED import build_CED, Block

# fusion module
from .modeling.fusion_module.AVFuse import AVFuse
from .utils.misc import channel_weighted_block
import matplotlib.pyplot as plt
import numpy as np

@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable # get input arguement from 'from_config'
    def __init__(
        self,
        *,
        backbone: Backbone,
        use_pre_sam: bool,  # * add pre sam
        pre_sam_backbone: Backbone,  # * add pre sam backbone
        scale_factor_module: nn.ModuleList,  # * add scale factor module
        audio_backbone: nn.Module,  # * add audio backbone
        audio_transformation: nn.Module or None,  # * add audio transformation
        sem_seg_head: nn.Module,
        fusion_module: nn.Module or None,  # * add fusion module
        criterion: nn.Module,
        is_avss_data: bool,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        audio_dim: int,
        frame_number: int,
        conv_dim: int,
        backbone_dim: list
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        # * Pre sam module
        self.use_pre_sam = use_pre_sam

        if self.use_pre_sam:
            self.pre_sam_backbone = pre_sam_backbone  # * add pre sam backbone
            self.scale_factor_module = scale_factor_module  # * add scale factor module
        else:
            self.pre_sam_backbone = None
        
        # * Audio module
        self.audio_backbone = audio_backbone  # * add audio backbone
        # * add audio transformation

        self.sem_seg_head = sem_seg_head

        # * Fusion module
        if fusion_module is not None:
            self.early_fusion = True
            self.fusion_module = fusion_module  # * add fusion module
            self.audio_transformation = audio_transformation
        else:
            self.early_fusion = False

        self.criterion = criterion
        self.is_avss_data = is_avss_data  # * add for avss data
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        self.visual_project = nn.ModuleList()
        self.time_visual_attn_blocks = nn.ModuleList()
        self.time_visual_norm = nn.ModuleList()
        self.time_attn_visual_feature = ['res2', 'res3', 'res4', 'res5']
        self.backbone_dim = backbone_dim

        for i in range(len(self.time_attn_visual_feature)):
            self.visual_project.append(nn.Sequential(
                        nn.Conv2d(self.backbone_dim[i], conv_dim, kernel_size=1),
                        nn.GroupNorm(32, conv_dim)))
            self.time_visual_attn_blocks.append(Block(dim = conv_dim, num_heads= 8, if_visual = True))
            self.time_visual_norm.append(nn.LayerNorm(conv_dim))
        
        self.time_attn_blocks_1 = Block(dim = audio_backbone.embed_dim, num_heads= 8)
        self.time_attn_blocks_2 = Block(dim = audio_backbone.embed_dim, num_heads= 8)
        self.time_attn_blocks_3 = Block(dim = audio_backbone.embed_dim, num_heads= 8)
        self.time_norm = nn.LayerNorm(audio_backbone.embed_dim)

        self.audio_linear1= nn.Linear(audio_backbone.embed_dim, audio_backbone.embed_dim)
        self.audio_linear2= nn.Linear(audio_backbone.embed_dim, audio_backbone.embed_dim)
        self.audio_linear3= nn.Linear(audio_backbone.embed_dim, audio_backbone.embed_dim)
        self.audio_linear4= nn.Linear(audio_backbone.embed_dim, audio_backbone.embed_dim)

        # self.frame_number = frame_number
        self.frame_number = 5 # always 5 for training
        
    @classmethod
    def from_config(cls, cfg):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # visual backbone

        backbone = build_backbone(cfg)
        # Whether to use pre sam

        use_pre_sam = cfg.MODEL.PRE_SAM.USE_PRE_SAM  # default: True
        pre_sam_dim = cfg.MODEL.PRE_SAM.PRE_SAM_DIM  # 

        if use_pre_sam:
            #* separate backbone
            pre_sam_backbone = build_backbone(cfg)  
            #* share backbone
            # pre_sam_backbone = backbone 
            scale_factor_module = nn.ModuleList()
            for dim in pre_sam_dim:
                scale_factor_module.append(channel_weighted_block(dim))
        else:
            pre_sam_backbone = None
            scale_factor_module = None

        audio_backbone = build_CED(device)

        # freeze audio backbone
        if cfg.MODEL.AUDIO.FREEZE_AUDIO_EXTRACTOR:
            for p in audio_backbone.parameters():
                p.requires_grad = False
            audio_backbone = FrozenBatchNorm2d.convert_frozen_batchnorm(audio_backbone)

        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # fusion module for early fusion
        if cfg.MODEL.FUSE_CONFIG.FUSION_STEP == "early":
            if cfg.MODEL.FUSE_CONFIG.QUERIES_FUSE_TYPE == "dim":
                audio_out_dim = 128  #  * cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES   #* 128 for dim concat 可调节
            else:
                audio_out_dim = 256  # * cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
            cfg.defrost()
            cfg.MODEL.FUSE_CONFIG.AUDIO_OUT_DIM = audio_out_dim  # * add audio out dim
            cfg.freeze()
            fusion_module = AVFuse(cfg)

        else:
            fusion_module = None
            audio_transformation = None

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        cosine_weight = cfg.MODEL.MASK_FORMER.COSINE_WEIGHT
        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
            "loss_cosine": cosine_weight,
        }  # add weight_dict for cosine loss
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        if cfg.INPUT.DATASET_MAPPER_NAME == "avss_semantic":
            is_avss_data = True
            criterion = SetCriterion_SS(
                sem_seg_head.num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            )
        else:
            is_avss_data = False
            criterion = SetCriterion(
                sem_seg_head.num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            )

        return {
            "backbone": backbone,
            "use_pre_sam": use_pre_sam,
            "pre_sam_backbone": pre_sam_backbone,  # * add pre sam backbone
            "audio_backbone": audio_backbone,  # * add audio backbone
            "scale_factor_module": scale_factor_module,  # * add scale factor module
            "audio_transformation": audio_transformation,  # * add audio transformation
            "sem_seg_head": sem_seg_head,
            "fusion_module": fusion_module,  # * add fusion module
            "criterion": criterion,
            "is_avss_data": is_avss_data,  # * add for avss data
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            'audio_dim': cfg.MODEL.FUSE_CONFIG.AUDIO_DIM,
            'frame_number': cfg.MODEL.FUSE_CONFIG.NUM_FRAMES,
            'conv_dim': cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
            'backbone_dim': cfg.MODEL.PRE_SAM.PRE_SAM_DIM
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """
        
        images = []
        pre_sam_masks = []
        audio_log_mels = None

        if self.is_avss_data:
            vid_temporal_mask_flag = None
            gt_temporal_mask_flag = None
            if self.training:
                data_type = []
                for batch_input in batched_inputs:
                    if batch_input["vid_temporal_mask_flag"].sum() == 10:
                          
                        random_index = random.randint(0, 5)

                        batch_input["vid_temporal_mask_flag"][0: random_index] = 0
                        batch_input["vid_temporal_mask_flag"][random_index +5 : ] = 0
                        index_ = torch.where(batch_input["vid_temporal_mask_flag"] == 1)[0]

                        data_type.append(index_)

                        batch_input["gt_temporal_mask_flag"] = batch_input["gt_temporal_mask_flag"][0:5]
                        

                        if vid_temporal_mask_flag == None:
                            vid_temporal_mask_flag = batch_input["vid_temporal_mask_flag"]
                        else:
                            vid_temporal_mask_flag = torch.cat((vid_temporal_mask_flag, batch_input["vid_temporal_mask_flag"]), dim=0)

                        if gt_temporal_mask_flag == None:
                            gt_temporal_mask_flag = batch_input["gt_temporal_mask_flag"]
                        else:
                            gt_temporal_mask_flag = torch.cat((gt_temporal_mask_flag, batch_input["gt_temporal_mask_flag"]), dim=0)
                        
                        images.extend(torch.unbind(batch_input["images"][batch_input["vid_temporal_mask_flag"].bool()].to(self.device), dim=0))
        
                        if self.use_pre_sam:
                            pre_sam_masks.extend(torch.unbind(batch_input["pre_masks"][batch_input["vid_temporal_mask_flag"].bool()].to(self.device), dim=0)) 

                        audio_log_mels = (
                            batch_input["audio_log_mel"].to(self.device)
                            if audio_log_mels is None
                            else torch.cat((audio_log_mels, batch_input["audio_log_mel"].to(self.device)), dim=0)
                        )
  
                    else:
                        data_type.append(0)
                        if vid_temporal_mask_flag == None:
                            vid_temporal_mask_flag = batch_input["vid_temporal_mask_flag"]
                        else:
                            vid_temporal_mask_flag = torch.cat((vid_temporal_mask_flag, batch_input["vid_temporal_mask_flag"]), dim=0)

                        if gt_temporal_mask_flag == None:
                            gt_temporal_mask_flag = batch_input["gt_temporal_mask_flag"]
                        else:
                            gt_temporal_mask_flag = torch.cat((gt_temporal_mask_flag, batch_input["gt_temporal_mask_flag"]), dim=0)
                        
                        images.extend(torch.unbind(batch_input["images"].to(self.device), dim=0))
        
                        if self.use_pre_sam:
                            pre_sam_masks.extend(torch.unbind(batch_input["pre_masks"].to(self.device), dim=0)) 

                        audio_log_mels = (
                            batch_input["audio_log_mel"].to(self.device)
                            if audio_log_mels is None
                            else torch.cat((audio_log_mels, batch_input["audio_log_mel"].to(self.device)), dim=0)
                        )
            else:
                for batch_input in batched_inputs:

                    if vid_temporal_mask_flag == None:
                        vid_temporal_mask_flag = batch_input["vid_temporal_mask_flag"]
                    else:
                        vid_temporal_mask_flag = torch.cat((vid_temporal_mask_flag, batch_input["vid_temporal_mask_flag"]), dim=0)
                    if gt_temporal_mask_flag == None:
                        gt_temporal_mask_flag = batch_input["gt_temporal_mask_flag"]
                    else:
                        gt_temporal_mask_flag = torch.cat((gt_temporal_mask_flag, batch_input["gt_temporal_mask_flag"]), dim=0)

                    images.extend(torch.unbind(batch_input["images"].to(self.device), dim=0))
        
                    if self.use_pre_sam:
                        pre_sam_masks.extend(torch.unbind(batch_input["pre_masks"].to(self.device), dim=0)) 

                    audio_log_mels = (
                        batch_input["audio_log_mel"].to(self.device)
                        if audio_log_mels is None
                        else torch.cat((audio_log_mels, batch_input["audio_log_mel"].to(self.device)), dim=0)
                    ) 

            vid_temporal_mask_flag = vid_temporal_mask_flag.to(self.device)
            gt_temporal_mask_flag = gt_temporal_mask_flag.to(self.device)

        else:
            for x in batched_inputs:

                images.extend(torch.unbind(x["images"].to(self.device), dim=0))

                if self.use_pre_sam:
                    pre_sam_masks.extend(torch.unbind(x["pre_masks"].to(self.device), dim=0))  # * [bs,1,224,224]

                audio_log_mels = (
                    x["audio_log_mel"].to(self.device)
                    if audio_log_mels is None
                    else torch.cat((audio_log_mels, x["audio_log_mel"].to(self.device)), dim=0)
                )  # * [bs*5, 1, 96, 64]
 
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)


        audio_log_mels = audio_log_mels.transpose(-1, -2)

        if self.is_avss_data:
            audio_log_mels = audio_log_mels[vid_temporal_mask_flag.bool()]

        with torch.no_grad():
            audio_feature = self.audio_backbone(audio_log_mels.squeeze(1)) 

        if not self.training:
            if self.is_avss_data:
                self.frame_number = audio_feature.shape[0]
            else:
                self.frame_number = 5      

        audio_list = []
        audio_list.append(self.audio_linear1(audio_feature))
        bt, n_p, c = audio_feature.shape
        bs = bt // self.frame_number

        audio_feature_t = audio_feature.contiguous().view(bs, self.frame_number, n_p, c)
        audio_feature_t = \
            audio_feature_t.permute(0, 2, 1, 3)\
        .contiguous().view(bs * n_p, self.frame_number, c)
        
        audio_feature = self.time_attn_blocks_1(audio_feature_t, if_return_audio = False)
        audio_feature, a_f1 = self.time_attn_blocks_2(audio_feature, if_return_audio = True)
        a_f1 = a_f1.contiguous().view(bs, n_p, self.frame_number, c)
        a_f1 = a_f1.transpose(1, 2).contiguous().view(bt, n_p, c)
        audio_list.append(self.audio_linear2(a_f1))
        audio_feature, a_f2, attn_map = self.time_attn_blocks_3(audio_feature, if_return_audio = True, if_return_attn_map = True)
        a_f2 = a_f2.contiguous().view(bs, n_p, self.frame_number, c)
        a_f2 = a_f2.transpose(1, 2).contiguous().view(bt, n_p, c)
        audio_list.append(self.audio_linear3(a_f2))
        audio_feature = self.time_norm(audio_feature)
        audio_feature = audio_feature.contiguous().view(bs, n_p, self.frame_number, c)
        audio_feature = audio_feature.transpose(1, 2).contiguous().view(bt, n_p, c)
        audio_list.append(self.audio_linear4(audio_feature))

        audio_feature = torch.cat(audio_list, dim = 1) 
        features = self.backbone(
            images.tensor
        )  

        if self.use_pre_sam:
            pre_sam_masks = [(x - self.pixel_mean) / self.pixel_std for x in pre_sam_masks]
            pre_sam_masks = ImageList.from_tensors(pre_sam_masks, self.size_divisibility)

            pre_sam_features = self.pre_sam_backbone(
                pre_sam_masks.tensor
            )  

            pre_sam_features_scale = [
                scale_factor_module(pre_sam_features[pre_sam_feature_key])
                for pre_sam_feature_key, scale_factor_module in zip(pre_sam_features.keys(), self.scale_factor_module)
            ]

            for i, key in enumerate(features.keys()):
                
                features[key] = features[key] + pre_sam_features_scale[i] * pre_sam_features[key]  #! wscale

        _, n_heads, _, _ = attn_map.shape
        attn_map = attn_map.contiguous().view(bs, n_p, n_heads, self.frame_number, self.frame_number)
        attn_map_audio = attn_map.mean(1).unsqueeze(1)

        for idx, name in enumerate(self.time_attn_visual_feature):
            attn_visual_feature = features[name]
            attn_visual_feature = self.visual_project[idx](attn_visual_feature)
            _, c_vis, attn_h, attn_w = attn_visual_feature.shape
            # print('attn_visual_feature', attn_visual_feature.shape)

            attn_map = attn_map_audio.repeat(1, attn_h * attn_w, 1, 1, 1)
            attn_map = attn_map.contiguous().view(bs * attn_h * attn_w, n_heads, self.frame_number, self.frame_number)

            attn_visual_feature = attn_visual_feature.contiguous().view(bs, self.frame_number, c_vis, attn_h, attn_w)
            attn_visual_feature = attn_visual_feature.permute(0, 3, 4, 1, 2)\
            .contiguous().view(bs * attn_h * attn_w, self.frame_number, c_vis)

            attn_visual_feature = self.time_visual_attn_blocks[idx](attn_visual_feature, input_attn = attn_map)
            attn_visual_feature = self.time_visual_norm[idx](attn_visual_feature)

            attn_visual_feature = attn_visual_feature.contiguous().view(bs, attn_h, attn_w, self.frame_number, c_vis)\
            .permute(0, 3, 4, 1, 2).contiguous().view(bt, c_vis, attn_h, attn_w)

            features[name] = attn_visual_feature
        
        # * Add fusion module here
        if self.early_fusion:
            # * early fusion
            fusion_feature = self.fusion_module(features, audio_feature)

            fusion_visual_feature = fusion_feature["visual"]
            fusion_audio_feature = fusion_feature["audio"]

            # do not use audio_transformation
            fusion_audio_feature = self.audio_transformation(fusion_audio_feature)  # * [bs*5, 256*N]

            outputs = self.sem_seg_head(fusion_visual_feature, fusion_audio_feature)  # dict {pred_logits, pred_masks, aux_outputs}
            
        else:
            outputs = self.sem_seg_head(features, audio_feature)  # dict {pred_logits, pred_masks, aux_outputs}

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = []
                # print('len(batched_inputs)', len(batched_inputs))
                if self.is_avss_data:
                    for i, x in enumerate(batched_inputs):
                        if isinstance(data_type[i], torch.Tensor):
                            # print('type(x["instances"])', type(x["instances"]))
                            x["instances"] = [x["instances"][index_instances] for index_instances in data_type[i]]
                        # print('len(x["instances"])', len(x["instances"]))
                        for gt_instance in x["instances"]:
                            # print('mask_shape', gt_instance.gt_masks.shape)
                            # print('gt_classes', gt_instance.gt_classes.shape)
                            gt_instance = gt_instance.to(self.device)
                            gt_instances.append(gt_instance)
                else:
                    for x in batched_inputs:
                        for gt_instance in x["instances"]:
                            # print('gt_instance', gt_instance.shape)
                            gt_instance = gt_instance.to(self.device)
                            gt_instances.append(gt_instance)
                # gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None
                raise ValueError("MaskFormer requires `instances` in training!")
            # bipartite matching-based loss
            if self.is_avss_data:
                losses = self.criterion(outputs, targets, vid_temporal_mask_flag, gt_temporal_mask_flag)
            else:
                losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
                    raise ValueError(f"Found useless Loss! {k}")
            return losses

        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            mid_feat = torch.stack(outputs["middles_attn_mask"], dim = 0)
            mid_feat = mid_feat.mean(0)

            frame_number = mid_feat.shape[0]
            mid_feat = mid_feat.reshape(frame_number, -1)

            sim_matrix = np.zeros([frame_number,frame_number])
            sim_matrix_color = np.zeros([frame_number,frame_number, 4])
            for i in range(frame_number):
                for j in range(frame_number):
                    # print(mid_feat[i].shape)
                    cos_sim = F.cosine_similarity(mid_feat[i], mid_feat[j], dim = 0)
                    cos_sim = cos_sim.item()
                    # color = plt.get_cmap('rainbow')(cos_sim)
                    # print(color)
                    # sim_matrix[i, j, :] = color
                    if i == j:
                        sim_matrix[i, j] = 1
                    else:
                        sim_matrix[i, j] = cos_sim

            matrix_round = np.round(sim_matrix, 5)
            sim_hist = list(matrix_round.reshape(-1))

            for i in range(frame_number):
                for j in range(frame_number):
                    color = plt.get_cmap('rainbow')(sim_matrix[i, j])
                    sim_matrix_color[i, j, :] = color

            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs
            num_video = -1
            processed_results = []
            if self.is_avss_data:
                self.num_frames = vid_temporal_mask_flag.sum()
            else:
                self.num_frames = 5
                
            for num_img, (mask_cls_result, mask_pred_result, image_size) in enumerate(
                zip(mask_cls_results, mask_pred_results, images.image_sizes)
            ):
                if num_img % self.num_frames == 0:
                    num_video += 1
                    input_per_image = batched_inputs[num_video]  

                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(mask_pred_result, image_size, height, width)
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                # retry_if_cuda_oom: Makes a function retry itself after encountering
                # pytorch's CUDA OOM error.
                if self.semantic_on:
                    if self.is_avss_data:
                        r = retry_if_cuda_oom(self.semantic_inference_ss)(mask_cls_result, mask_pred_result, vid_temporal_mask_flag[num_img])
                    else:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        # sem_seg_postprocess：Return semantic segmentation predictions in the original resolution.
                    processed_results[-1]["sem_seg"] = r
                    # processed_results[-1]["attn_map"] = return_attn_map[num_img]
                # TODO for visualization
                # visual_middle_features = True
                # if visual_middle_features:
                #     return processed_results, mask_pred_results
            return processed_results, sim_matrix_color, sim_hist

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def semantic_inference_ss(self, mask_cls, mask_pred, vid_temporal_mask_flag):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        semseg = semseg * vid_temporal_mask_flag
        return semseg

    def mul_temporal_mask(self, feats, vid_temporal_mask_flag):
        """
        Args:
            feats: [bs*10, C, H, W]
            vid_temporal_mask_flag: [bs*10, 1, 1, 1]
        """
        out = feats * vid_temporal_mask_flag
        return out
