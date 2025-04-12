import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.init import normal_
from torch.nn import L1Loss, BCEWithLogitsLoss, SmoothL1Loss
from fvcore.nn import sigmoid_focal_loss_jit
import math

from models.sparse_ins import SparseInsDecoder
from .utils import inverse_sigmoid
from .transformer_bricks import *
from utils.utils import multi_apply


class LATRHead(nn.Module):
    def __init__(self, args,
                 dim=128,
                 num_group=1,
                 num_convs=4,
                 in_channels=128,
                 kernel_dim=128,
                 num_classes=21,
                 num_query=30,
                 embed_dims=128,
                 num_reg_fcs=2,
                 depth_num=50,
                 depth_start=3,
                 top_view_region=None,
                 position_range=[-50, 3, -10, 50, 103, 10.],
                 pred_dim=10,
                 xs_loss_weight=1.0,
                 zs_loss_weight=5.0,
                 vis_loss_weight=1.0,
                 cls_loss_weight=0.0001,
                 project_loss_weight=1.0,
                 trans_params=dict(
                     init_z=0, bev_h=250, bev_w=100),
                 pt_as_query=False,
                 num_pt_per_line=20,
                 num_feature_levels=1,
                 gt_project_h=20,
                 gt_project_w=30,
                 project_crit=dict(
                     type='SmoothL1Loss',
                     reduction='none'),
                 ):
        super().__init__()
        self.trans_params = dict(
            top_view_region=top_view_region,
            z_region=[position_range[2], position_range[5]])
        self.trans_params.update(trans_params)
        self.gt_project_h = gt_project_h
        self.gt_project_w = gt_project_w
        self.num_y_steps = args.num_y_steps
        self.register_buffer('anchor_y_steps',
            torch.from_numpy(args.anchor_y_steps).float())
        self.register_buffer('anchor_y_steps_dense',
            torch.from_numpy(args.anchor_y_steps_dense).float())
        
        project_crit['reduction'] = 'none'
        self.project_crit = SmoothL1Loss(
            beta=1.0,
            reduction='none')
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.code_size = pred_dim
        self.num_query = num_query
        self.num_group = num_group
        self.num_pred = args.num_decoder_layers
        self.pc_range = position_range
        self.xs_loss_weight = xs_loss_weight
        self.zs_loss_weight = zs_loss_weight
        self.vis_loss_weight = vis_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.project_loss_weight = project_loss_weight

        self.reg_crit = L1Loss(
            reduction='mean')
        # self.cls_crit = sigmoid_focal_loss_jit # just directly use focal loss in the forward function, can without init
        self.bce_loss = BCEWithLogitsLoss(reduction='mean')

        self.sparse_ins = SparseInsDecoder(args)

        self.depth_num = depth_num
        self.position_dim = 3 * self.depth_num
        self.position_range = position_range
        self.depth_start = depth_start
        self.adapt_pos3d = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )
        self.positional_encoding = SinePositionalEncoding(num_feats=256 // 2, normalize=True)
        self.position_encoder = nn.Sequential(
            nn.Conv2d(self.position_dim, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )
        self.transformer = LATRTransformer(args)
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        # build pred layer: cls, reg, vis
        self.num_reg_fcs = num_reg_fcs
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.num_classes))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(
            nn.Linear(
                self.embed_dims,
                3 * self.code_size // num_pt_per_line))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

        self.num_pt_per_line = num_pt_per_line
        self.point_embedding = nn.Embedding(
            self.num_pt_per_line, self.embed_dims)

        self.reference_points = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(True),
            nn.Linear(self.embed_dims, 2 * self.code_size // num_pt_per_line))
        self.num_feature_levels = num_feature_levels
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))

        self._init_weights()

    def _init_weights(self):
        self.transformer.init_weights()
        # nn.init.xavier_uniform_(self.reference_points)
        normal_(self.level_embeds)

    def forward(self, input_dict, is_training=True):
        output_dict = {}
        img_feats = input_dict['x']

        if not isinstance(img_feats, (list, tuple)):
            img_feats = [img_feats]

        sparse_output = self.sparse_ins(
            img_feats[0],
            lane_idx_map=input_dict['lane_idx'],
            input_shape=input_dict['seg'].shape[-2:],
            is_training=is_training)
        # generate 2d pos emb
        B, C, H, W = img_feats[0].shape
        masks = img_feats[0].new_zeros((B, H, W))

        # TODO use actual mask if using padding or other aug
        sin_embed = self.positional_encoding(masks)
        sin_embed = self.adapt_pos3d(sin_embed)

        # init query and reference pt
        query = sparse_output['inst_features'] # BxNxC
        # B, N, C -> B, N, num_anchor_per_line, C
        query = query.unsqueeze(2) + self.point_embedding.weight[None, None, ...]

        query_embeds = self.query_embedding(query).flatten(1, 2)
        query = torch.zeros_like(query_embeds)
        reference_points = self.reference_points(query_embeds)
        reference_points = reference_points.sigmoid()
        mlvl_feats = img_feats

        feat_flatten = []
        spatial_shapes = []
        mlvl_masks = []

        assert self.num_feature_levels == len(mlvl_feats)
        for lvl, feat in enumerate(mlvl_feats):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(2).permute(2, 0, 1) # NxBxC
            feat = feat + self.level_embeds[None, lvl:lvl+1, :].to(feat.device)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
            mlvl_masks.append(torch.zeros((bs, *spatial_shape),
                                           dtype=torch.bool,
                                           device=feat.device))

        if self.transformer.with_encoder:
            mlvl_positional_encodings = []
            pos_embed2d = []
            for lvl, feat in enumerate(mlvl_feats):
                mlvl_positional_encodings.append(
                    self.positional_encoding(mlvl_masks[lvl]))
                pos_embed2d.append(
                    mlvl_positional_encodings[-1].flatten(2).permute(2, 0, 1))
            pos_embed2d = torch.cat(pos_embed2d, 0)
        else:
            mlvl_positional_encodings = None
            pos_embed2d = None

        feat_flatten = torch.cat(feat_flatten, 0)

        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=query.device)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1, )),
             spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        # head
        pos_embed = None
        outs_dec, project_results, outputs_classes, outputs_coords = \
            self.transformer(
                feat_flatten, None,
                query, query_embeds, pos_embed,
                reference_points=reference_points,
                reg_branches=self.reg_branches,
                cls_branches=self.cls_branches,
                img_feats=img_feats,
                lidar2img=input_dict['lidar2img'],
                pad_shape=input_dict['pad_shape'],
                sin_embed=sin_embed,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                mlvl_masks=mlvl_masks,
                mlvl_positional_encodings=mlvl_positional_encodings,
                pos_embed2d=pos_embed2d,
                image=input_dict['image'],
                **self.trans_params)

        all_cls_scores = torch.stack(outputs_classes)
        all_line_preds = torch.stack(outputs_coords)
        all_line_preds[..., 0] = (all_line_preds[..., 0]
            * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        all_line_preds[..., 1] = (all_line_preds[..., 1]
            * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

        # reshape to original format
        all_line_preds = all_line_preds.view(
            len(outputs_classes), bs, self.num_query,
            self.transformer.decoder.num_anchor_per_query,
            self.transformer.decoder.num_points_per_anchor, 2 + 1 # xz+vis
        )
        all_line_preds = all_line_preds.permute(0, 1, 2, 5, 3, 4)
        all_line_preds = all_line_preds.flatten(3, 5)

        output_dict.update({
            'all_cls_scores': all_cls_scores,
            'all_line_preds': all_line_preds,
        })
        output_dict.update(sparse_output)

        if is_training:
            losses = self.get_loss(output_dict, input_dict)
            project_loss = self.get_project_loss(
                project_results, input_dict,
                h=self.gt_project_h, w=self.gt_project_w)
            losses['project_loss'] = \
                self.project_loss_weight * project_loss
            output_dict.update(losses)
        return output_dict

    def get_project_loss(self, results, input_dict, h=20, w=30):
        gt_lane = input_dict['ground_lanes_dense']
        gt_ys = self.anchor_y_steps_dense.clone()
        code_size = gt_ys.shape[0]
        gt_xs = gt_lane[..., :code_size]
        gt_zs = gt_lane[..., code_size : 2*code_size]
        gt_vis = gt_lane[..., 2*code_size:3*code_size]
        gt_ys = gt_ys[None, None, :].expand_as(gt_xs)
        gt_points = torch.stack([gt_xs, gt_ys, gt_zs], dim=-1)

        B = results[0].shape[0]
        ref_3d_home = F.pad(gt_points, (0, 1), value=1)
        coords_img = ground2img(
            ref_3d_home,
            h, w,
            input_dict['lidar2img'],
            input_dict['pad_shape'], mask=gt_vis)

        all_loss = 0.
        for projct_result in results:
            projct_result = F.interpolate(
                projct_result,
                size=(h, w),
                mode='nearest')
            gt_proj = coords_img.clone()

            mask = (gt_proj[:, -1, ...] > 0) * (projct_result[:, -1, ...] > 0)
            diff_loss = self.project_crit(
                projct_result[:, :3, ...],
                gt_proj[:, :3, ...],
            )
            diff_y_loss = diff_loss[:, 1, ...]
            diff_z_loss = diff_loss[:, 2, ...]
            diff_loss = diff_y_loss * 0.1 + diff_z_loss
            diff_loss = (diff_loss * mask).sum() / torch.clamp(mask.sum(), 1)
            all_loss = all_loss + diff_loss

        return all_loss / len(results)

    def get_loss(self, output_dict, input_dict):
        all_cls_pred = output_dict['all_cls_scores']
        all_lane_pred = output_dict['all_line_preds']
        gt_lanes = input_dict['ground_lanes']
        all_xs_loss = 0.0
        all_zs_loss = 0.0
        all_vis_loss = 0.0
        all_cls_loss = 0.0
        matched_indices = output_dict['matched_indices']
        num_layers = all_lane_pred.shape[0]

        def single_layer_loss(layer_idx):
            gcls_pred = all_cls_pred[layer_idx]
            glane_pred = all_lane_pred[layer_idx]

            glane_pred = glane_pred.view(
                glane_pred.shape[0],
                self.num_group,
                self.num_query,
                glane_pred.shape[-1])
            gcls_pred = gcls_pred.view(
                gcls_pred.shape[0],
                self.num_group,
                self.num_query,
                gcls_pred.shape[-1])

            per_xs_loss = 0.0
            per_zs_loss = 0.0
            per_vis_loss = 0.0
            per_cls_loss = 0.0
            batch_size = len(matched_indices[0])

            for b_idx in range(len(matched_indices[0])):
                for group_idx in range(self.num_group):
                    pred_idx = matched_indices[group_idx][b_idx][0]
                    gt_idx = matched_indices[group_idx][b_idx][1]

                    cls_pred = gcls_pred[:, group_idx, ...]
                    lane_pred = glane_pred[:, group_idx, ...]

                    if gt_idx.shape[0] < 1:
                        cls_target = cls_pred.new_zeros(cls_pred[b_idx].shape[0]).long()
                        cls_loss = sigmoid_focal_loss_jit(cls_pred[b_idx], cls_target, alpha=0.25, gamma=2.0, reduction="mean")
                        per_cls_loss = per_cls_loss + cls_loss
                        per_xs_loss = per_xs_loss + 0.0 * lane_pred[b_idx].mean()
                        continue

                    pos_lane_pred = lane_pred[b_idx][pred_idx]
                    gt_lane = gt_lanes[b_idx][gt_idx]

                    pred_xs = pos_lane_pred[:, :self.code_size]
                    pred_zs = pos_lane_pred[:, self.code_size : 2*self.code_size]
                    pred_vis = pos_lane_pred[:, 2*self.code_size:]
                    gt_xs = gt_lane[:, :self.code_size]
                    gt_zs = gt_lane[:, self.code_size : 2*self.code_size]
                    gt_vis = gt_lane[:, 2*self.code_size:3*self.code_size]

                    loc_mask = gt_vis > 0
                    xs_loss = self.reg_crit(pred_xs, gt_xs)
                    zs_loss = self.reg_crit(pred_zs, gt_zs)
                    xs_loss = (xs_loss * loc_mask).sum() / torch.clamp(loc_mask.sum(), 1)
                    zs_loss = (zs_loss * loc_mask).sum() / torch.clamp(loc_mask.sum(), 1)
                    vis_loss = self.bce_loss(pred_vis, gt_vis)

                    cls_target = cls_pred.new_zeros(cls_pred[b_idx].shape[0]).long()
                    cls_target[pred_idx] = torch.argmax(
                        gt_lane[:, 3*self.code_size:], dim=1)
                    cls_pred_from_one_hot = torch.argmax(cls_pred[b_idx], dim=1)
                    cls_loss = sigmoid_focal_loss_jit(cls_pred_from_one_hot, cls_target, alpha=0.25, gamma=2.0, reduction="mean")

                    per_xs_loss += xs_loss
                    per_zs_loss += zs_loss
                    per_vis_loss += vis_loss
                    per_cls_loss += cls_loss

            return tuple(map(lambda x: x / batch_size / self.num_group,
                             [per_xs_loss, per_zs_loss, per_vis_loss, per_cls_loss]))

        all_xs_loss, all_zs_loss, all_vis_loss, all_cls_loss = multi_apply(
            single_layer_loss, range(all_lane_pred.shape[0]))
        all_xs_loss = sum(all_xs_loss) / num_layers
        all_zs_loss = sum(all_zs_loss) / num_layers
        all_vis_loss = sum(all_vis_loss) / num_layers
        all_cls_loss = sum(all_cls_loss) / num_layers

        return dict(
            all_xs_loss=self.xs_loss_weight * all_xs_loss,
            all_zs_loss=self.zs_loss_weight * all_zs_loss,
            all_vis_loss=self.vis_loss_weight * all_vis_loss,
            all_cls_loss=self.cls_loss_weight * all_cls_loss,
        )

    @staticmethod
    def get_reference_points(H, W, bs=1, device='cuda', dtype=torch.float):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(
                0.5, W - 0.5, W, dtype=dtype, device=device)
        )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1) 
        return ref_2d






class SinePositionalEncoding(nn.Module):
    def __init__(self, num_feats, temperature=10000, normalize=False, scale=2*math.pi, eps=1e-6, offset=0.):
        super().__init__()
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask):
        mask = mask.to(torch.int)
        not_mask = 1- mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, H, W = mask.size()
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos