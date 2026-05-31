import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class CAPRelationMixer(torch.nn.Module):
    """
    CAP relation module that mixes local multi-scale context with global token
    interactions across fixed RoI bins.
    """
    def __init__(self, embed_dim, roi_size, dropout_ratio):
        super().__init__()
        token_hidden = max(roi_size * 2, 16)
        gate_hidden = max(embed_dim // 4, 32)

        self.local_norm = nn.LayerNorm(embed_dim)
        self.local_dw3 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.local_dw5 = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2, groups=embed_dim)
        self.local_pw = nn.Conv1d(embed_dim * 2, embed_dim, kernel_size=1)

        self.token_norm = nn.LayerNorm(embed_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(roi_size, token_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(token_hidden, roi_size),
        )

        self.branch_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 2),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
        )

    def forward(self, feat):
        """
        Inputs:
            feat: tensor of size [N, roi_size, D]
        """
        local_feat = self.local_norm(feat).transpose(1, 2)                             # [N, D, roi]
        local_feat = self.local_pw(torch.cat((self.local_dw3(local_feat), self.local_dw5(local_feat)), dim=1))
        local_feat = local_feat.transpose(1, 2)                                         # [N, roi, D]

        token_feat = self.token_norm(feat).transpose(1, 2)                              # [N, D, roi]
        token_feat = self.token_mlp(token_feat).transpose(1, 2)                         # [N, roi, D]

        branch_summary = torch.cat((local_feat.mean(dim=1), token_feat.mean(dim=1)), dim=-1)
        branch_weight = F.softmax(self.branch_gate(branch_summary), dim=-1)             # [N, 2]
        local_weight = branch_weight[:, [0]].unsqueeze(2)                               # [N, 1, 1]
        token_weight = branch_weight[:, [1]].unsqueeze(2)                               # [N, 1, 1]
        mixed_feat = local_weight * local_feat + token_weight * token_feat

        return feat + self.out_proj(mixed_feat)


class CrossProposalMemory(torch.nn.Module):
    """
    Proposal-level Cross-video Support Memory.

    The module relies on positive cross-video supports, edits proposal scores
    with retrieved class-wise support evidence, and suppresses absent classes via
    class competition on the same pseudo-positive anchors.
    """
    def __init__(self, input_dim, n_class, dropout_ratio, embed_dim=256, bank_size=64, topk=4, temperature=0.07):
        super().__init__()
        self.n_class = n_class
        self.embed_dim = embed_dim
        self.bank_size = bank_size
        self.topk = topk
        self.temperature = temperature

        hidden_dim = max(embed_dim, 128)
        self.query_proj = nn.Sequential(
            nn.Linear(input_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.context_pair = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.context_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
        )
        self.target_query_proj = copy.deepcopy(self.query_proj)
        self.target_context_pair = copy.deepcopy(self.context_pair)
        self.target_context_out = copy.deepcopy(self.context_out)
        for module in (self.target_query_proj, self.target_context_pair, self.target_context_out):
            for param in module.parameters():
                param.requires_grad = False
        self.rule_encoder = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.edit_head = nn.Linear(32, 2)
        self.reliability_head = nn.Linear(32, 1)

        self.register_buffer('pos_memory_feat', torch.zeros(n_class, bank_size, embed_dim))
        self.register_buffer('pos_memory_span', torch.zeros(n_class, bank_size))
        self.register_buffer('pos_memory_conf', torch.zeros(n_class, bank_size))
        self.register_buffer('pos_memory_valid', torch.zeros(n_class, bank_size, dtype=torch.bool))
        self.register_buffer('pos_memory_video', torch.full((n_class, bank_size), -1, dtype=torch.long))
        self.register_buffer('pos_memory_ptr', torch.zeros(n_class, dtype=torch.long))

    def _segments_iou(self, segments):
        start = segments[..., 0]
        end = segments[..., 1]
        inter_start = torch.maximum(start.unsqueeze(-1), start.unsqueeze(-2))
        inter_end = torch.minimum(end.unsqueeze(-1), end.unsqueeze(-2))
        inter = (inter_end - inter_start).clamp(min=0.0)
        duration = (end - start).clamp(min=1e-4)
        union = duration.unsqueeze(-1) + duration.unsqueeze(-2) - inter
        return inter / (union + 1e-6)

    def _contextualize_queries(self, query_feat, prop_segments, prop_scalar_conf, prop_mask, use_target=False):
        valid_mask = prop_mask.bool()
        start = prop_segments[..., 0]
        end = prop_segments[..., 1]
        center = 0.5 * (start + end)
        duration = (end - start).clamp(min=1e-4)
        center_gap = (center.unsqueeze(2) - center.unsqueeze(1)).abs()
        duration_gap = (torch.log(duration).unsqueeze(2) - torch.log(duration).unsqueeze(1)).abs()
        pair_feat = torch.stack((
            self._segments_iou(prop_segments),
            -center_gap,
            -duration_gap,
        ), dim=-1)
        context_pair = self.target_context_pair if use_target else self.context_pair
        context_out = self.target_context_out if use_target else self.context_out
        pair_logit = context_pair(pair_feat).squeeze(-1)
        invalid = ~(valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2))
        pair_logit = pair_logit.masked_fill(invalid, -1e4)
        pair_weight = F.softmax(pair_logit, dim=-1)
        pair_weight = pair_weight * valid_mask.unsqueeze(1).float()
        pair_weight = pair_weight / (pair_weight.sum(dim=-1, keepdim=True) + 1e-6)

        context_feat = torch.matmul(pair_weight, query_feat)
        context_conf = (pair_weight * prop_scalar_conf.squeeze(-1).unsqueeze(1)).sum(dim=-1, keepdim=True)
        contextualized = F.normalize(query_feat + context_out(context_feat), dim=-1)
        return contextualized, context_conf

    def encode_queries(self, prop_repr, prop_segments, feature_lengths, prop_span, prop_scalar_conf, prop_action_prob, prop_mask, use_target=False):
        query_input = torch.cat((prop_repr, prop_span, prop_scalar_conf, prop_action_prob), dim=-1)
        query_proj = self.target_query_proj if use_target else self.query_proj
        query_feat = F.normalize(query_proj(query_input), dim=-1)
        norm_segments = prop_segments / feature_lengths.unsqueeze(1).unsqueeze(-1).clamp(min=1e-4)
        query_feat, context_conf = self._contextualize_queries(query_feat, norm_segments, prop_scalar_conf, prop_mask, use_target=use_target)
        return query_feat, context_conf

    def _retrieve_bank_stats(self, query_flat, span_flat, query_video_flat, memory_feat, memory_span, memory_conf, memory_valid, memory_video):
        align = query_flat.new_zeros((query_flat.shape[0], self.n_class))
        mean = query_flat.new_zeros((query_flat.shape[0], self.n_class))
        conf = query_flat.new_zeros((query_flat.shape[0], self.n_class))
        span_gap = query_flat.new_ones((query_flat.shape[0], self.n_class))
        valid = query_flat.new_zeros((query_flat.shape[0], self.n_class))

        for c in range(self.n_class):
            valid_idx = torch.where(memory_valid[c])[0]
            if valid_idx.numel() == 0:
                continue

            memory_feat_c = memory_feat[c, valid_idx]
            memory_conf_c = memory_conf[c, valid_idx]
            memory_span_c = memory_span[c, valid_idx]
            memory_video_c = memory_video[c, valid_idx]
            sim_c = torch.matmul(query_flat, memory_feat_c.transpose(0, 1))
            if torch.any(query_video_flat >= 0):
                same_video = query_video_flat.unsqueeze(1) == memory_video_c.unsqueeze(0)
                sim_c = sim_c.masked_fill(same_video, -1e4)
            k = min(self.topk, int(valid_idx.numel()))
            top_val, top_idx = torch.topk(sim_c, k=k, dim=1)
            top_mask = top_val > -1e3
            masked_top_val = top_val.masked_fill(~top_mask, -1e4)
            top_weight = F.softmax(masked_top_val / self.temperature, dim=1)
            top_weight = top_weight * top_mask.float()
            top_weight = top_weight / (top_weight.sum(dim=1, keepdim=True) + 1e-6)

            top_feat = memory_feat_c[top_idx]
            proto = (top_weight.unsqueeze(-1) * top_feat).sum(dim=1)
            proto = F.normalize(proto + 1e-6, dim=-1)
            valid_row = top_mask.any(dim=1).float()

            align[:, c] = (proto * query_flat).sum(dim=-1) * valid_row
            mean[:, c] = (top_weight * top_val.masked_fill(~top_mask, 0)).sum(dim=1)
            conf[:, c] = (top_weight * memory_conf_c[top_idx]).sum(dim=1)
            span_gap[:, c] = (span_flat - (top_weight * memory_span_c[top_idx]).sum(dim=1)).abs()
            valid[:, c] = valid_row

        return align, mean, conf, span_gap, valid

    def forward(self, prop_repr, prop_segments, feature_lengths, prop_span, prop_scalar_conf, prop_action_prob, prop_mask, query_video_ids=None):
        """
        Inputs:
            prop_repr: tensor of size [B, M, D]
            prop_segments: tensor of size [B, M, 2]
            feature_lengths: tensor of size [B]
            prop_span: tensor of size [B, M, 1]
            prop_scalar_conf: tensor of size [B, M, 1]
            prop_action_prob: tensor of size [B, M, 1]
            prop_mask: tensor of size [B, M]
        """
        query_feat, context_conf = self.encode_queries(
            prop_repr,
            prop_segments,
            feature_lengths,
            prop_span,
            prop_scalar_conf,
            prop_action_prob,
            prop_mask,
            use_target=False,
        )

        bsz, num_prop, _ = query_feat.shape
        query_flat = query_feat.reshape(bsz * num_prop, self.embed_dim)
        span_flat = prop_span.reshape(bsz * num_prop)
        conf_flat = prop_scalar_conf.reshape(bsz * num_prop)
        action_flat = prop_action_prob.reshape(bsz * num_prop)
        context_flat = context_conf.reshape(bsz * num_prop)
        if query_video_ids is None:
            query_video_flat = torch.full((bsz * num_prop,), -1, dtype=torch.long, device=query_feat.device)
        else:
            query_video_flat = query_video_ids.reshape(-1, 1).repeat(1, num_prop).reshape(-1).to(query_feat.device)

        pos_align, pos_mean, pos_conf, pos_span_gap, pos_valid = self._retrieve_bank_stats(
            query_flat,
            span_flat,
            query_video_flat,
            self.pos_memory_feat,
            self.pos_memory_span,
            self.pos_memory_conf,
            self.pos_memory_valid,
            self.pos_memory_video,
        )

        stat_input = torch.stack((
            pos_align,
            pos_mean,
            pos_conf,
            -pos_span_gap,
            conf_flat.unsqueeze(1).expand(-1, self.n_class),
            action_flat.unsqueeze(1).expand(-1, self.n_class),
            context_flat.unsqueeze(1).expand(-1, self.n_class),
        ), dim=-1)
        rule_feat = self.rule_encoder(stat_input)
        positive_valid = (pos_valid > 0).float()
        base_reliability = torch.sigmoid(self.reliability_head(rule_feat)).squeeze(-1) * positive_valid
        edit_raw = self.edit_head(rule_feat)
        support_score = 0.6 * pos_align + 0.4 * pos_mean
        cls_logit = edit_raw[..., 0] + 1.10 * support_score
        bonus_logit = edit_raw[..., 1] + 1.35 * support_score + 0.20 * pos_conf
        cpm_cls_delta = torch.tanh(cls_logit) * base_reliability
        cpm_bonus = torch.sigmoid(bonus_logit) * base_reliability * torch.sigmoid(1.5 * support_score)

        return {
            'positive_response': pos_align.reshape(bsz, num_prop, self.n_class),          # [B, M, C]
            'cpm_cls_delta': cpm_cls_delta.reshape(bsz, num_prop, self.n_class),          # [B, M, C]
            'cpm_bonus': cpm_bonus.reshape(bsz, num_prop, self.n_class),                  # [B, M, C]
            'cpm_pos_valid': pos_valid.reshape(bsz, num_prop, self.n_class),              # [B, M, C]
        }

    @torch.no_grad()
    def reset_memory(self):
        for name, buffer in self.named_buffers():
            if 'memory_' in name:
                if buffer.dtype == torch.bool:
                    buffer.zero_()
                elif buffer.dtype in (torch.int32, torch.int64, torch.long):
                    if 'video' in name:
                        buffer.fill_(-1)
                    else:
                        buffer.zero_()
                else:
                    buffer.zero_()

    @torch.no_grad()
    def update_target_encoder(self, momentum):
        source_target_pairs = (
            (self.query_proj, self.target_query_proj),
            (self.context_pair, self.target_context_pair),
            (self.context_out, self.target_context_out),
        )
        for source_module, target_module in source_target_pairs:
            for param_src, param_tgt in zip(source_module.parameters(), target_module.parameters()):
                param_tgt.data.mul_(momentum).add_(param_src.data, alpha=1.0 - momentum)

    def _push_memory(self, memory_feat, memory_span, memory_conf, memory_valid, memory_video, memory_ptr, class_idx, feat, span, conf, video_id):
        ptr = int(memory_ptr[class_idx].item())
        memory_feat[class_idx, ptr] = feat
        memory_span[class_idx, ptr] = span
        memory_conf[class_idx, ptr] = conf
        memory_valid[class_idx, ptr] = True
        memory_video[class_idx, ptr] = video_id
        memory_ptr[class_idx] = (ptr + 1) % self.bank_size

    @torch.no_grad()
    def update_memory(self, prop_repr, prop_segments, feature_lengths, prop_span, prop_scalar_conf, prop_action_prob, prop_mask, source_cas, labels, video_ids, args):
        query_feat, _ = self.encode_queries(
            prop_repr.detach(),
            prop_segments.detach(),
            feature_lengths.detach(),
            prop_span.detach(),
            prop_scalar_conf.detach(),
            prop_action_prob.detach(),
            prop_mask.detach(),
            use_target=True,
        )
        cls_prob = F.softmax(source_cas.detach(), dim=-1)[:, :, :self.n_class]
        seed_score = prop_scalar_conf.detach() * cls_prob
        valid_mask = prop_mask.bool()

        for b in range(query_feat.shape[0]):
            valid = valid_mask[b]
            if valid.sum() == 0:
                continue

            query_b = query_feat[b][valid]
            span_b = prop_span[b][valid].squeeze(-1)
            seed_b = seed_score[b][valid]
            video_id_b = int(video_ids[b].item())

            positive_cls = torch.where(labels[b] > 0)[0]
            for c in positive_cls.tolist():
                score_c = seed_b[:, c]
                if score_c.numel() == 0 or torch.max(score_c) <= 0:
                    continue
                retained = score_c > args.cpm_seed_gamma * torch.max(score_c)
                if retained.sum() == 0:
                    continue
                k = min(args.cpm_update_topk, int(retained.sum().item()))
                idx = torch.topk(score_c.masked_fill(~retained, -1e6), k=k, dim=0).indices
                for proposal_idx in idx.tolist():
                    self._push_memory(
                        self.pos_memory_feat,
                        self.pos_memory_span,
                        self.pos_memory_conf,
                        self.pos_memory_valid,
                        self.pos_memory_video,
                        self.pos_memory_ptr,
                        c,
                        query_b[proposal_idx],
                        span_b[proposal_idx],
                        score_c[proposal_idx],
                        video_id_b,
                    )


class Backbone_Proposal(torch.nn.Module):
    """
    Backbone for a single modality in P-MIL, including CAP.
    """
    def __init__(self, feat_dim, n_class, dropout_ratio, roi_size, gate_prior_alpha=0.0, rescue_context_alpha=0.5, rescue_iou_alpha=0.2, rescue_ambiguity_alpha=0.7):
        super().__init__()
        embed_dim = feat_dim // 2
        self.roi_size = roi_size
        self.edge_size = max(1, roi_size // 6)
        self.gate_prior_alpha = gate_prior_alpha
        self.rescue_context_alpha = rescue_context_alpha
        self.rescue_iou_alpha = rescue_iou_alpha
        self.rescue_ambiguity_alpha = rescue_ambiguity_alpha

        self.prop_fusion = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
        )
        self.prop_classifier = nn.Sequential(
            nn.Conv1d(feat_dim, embed_dim, 1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, n_class+1, 1),
        )
        self.prop_attention = nn.Sequential(
            nn.Conv1d(feat_dim, embed_dim, 1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, 1, 1),
        )
        self.prop_completeness = nn.Sequential(
            nn.Conv1d(feat_dim, embed_dim, 1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, 1, 1),
        )
        self.cap_embed = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
        )
        self.cap_mixer = CAPRelationMixer(embed_dim, roi_size, dropout_ratio)
        self.cap_classifier = nn.Linear(embed_dim, n_class + 1)
        self.cap_occupancy = nn.Linear(embed_dim, 1)
        self.cap_class_occupancy = nn.Linear(embed_dim, n_class + 1)
        self.cap_iou_occupancy = nn.Linear(embed_dim, 1)
        self.cap_cls_gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_class + 1),
        )
        self.cap_iou_gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )
        self.cap_delta_cls = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_class + 1),
        )
        self.cap_delta_iou = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def _aggregate_cap_logits(self, cap_logits, weights):
        weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-6)
        return (cap_logits * weights).sum(dim=2)

    def _aggregate_cap_features(self, cap_feat, weights):
        weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-6)
        return (cap_feat * weights).sum(dim=2)

    def forward(self, feat):
        """
        Inputs:
            feat: tensor of size [B, M, roi_size, D]

        Outputs:
            prop_cas:  tensor of size [B, C, M]
            prop_attn: tensor of size [B, 1, M]
            prop_iou:  tensor of size [B, 1, M]
        """
        feat_raw = feat
        feat1 = feat_raw[:, :,                   : self.edge_size  , :].max(2)[0]
        feat2 = feat_raw[:, :, self.edge_size    : self.roi_size - self.edge_size, :].max(2)[0]
        feat3 = feat_raw[:, :, self.roi_size - self.edge_size:                   , :].max(2)[0]
        feat = torch.cat((feat2-feat1, feat2, feat2-feat3), dim=2)

        feat_fuse = self.prop_fusion(feat)                              # [B, M, D]
        feat_fuse = feat_fuse.transpose(-1, -2)                         # [B, D, M]

        prop_cas = self.prop_classifier(feat_fuse)                      # [B, C, M]
        prop_attn = self.prop_attention(feat_fuse)                      # [B, 1, M]
        prop_iou = self.prop_completeness(feat_fuse)                    # [B, 1, M]
        cap_feat = self.cap_embed(feat_raw)                             # [B, M, roi_size, D/2]
        bsz, num_prop, num_roi_bin, hid_dim = cap_feat.shape
        cap_feat = self.cap_mixer(cap_feat.reshape(bsz * num_prop, num_roi_bin, hid_dim)).reshape(bsz, num_prop, num_roi_bin, hid_dim)

        cap_cas = self.cap_classifier(cap_feat)                         # [B, M, roi_size, C]
        cap_quality_logit = self.cap_occupancy(cap_feat)                # CAP quality
        cap_cls_occ = torch.sigmoid(self.cap_class_occupancy(cap_feat) + cap_quality_logit)
        cap_iou_occ = torch.sigmoid(self.cap_iou_occupancy(cap_feat) + cap_quality_logit)
        cap_cls_summary = cap_cls_occ[..., :-1].max(dim=-1, keepdim=True)[0]

        prop_core_cas = self._aggregate_cap_logits(cap_cas, cap_cls_occ)
        prop_context_cas = self._aggregate_cap_logits(cap_cas, 1 - cap_cls_occ)
        prop_dom_logit = prop_core_cas - prop_context_cas

        prop_cls_core_feat = self._aggregate_cap_features(cap_feat, cap_cls_summary)
        prop_cls_context_feat = self._aggregate_cap_features(cap_feat, 1 - cap_cls_summary)
        prop_cls_calib_feat = torch.cat((prop_cls_core_feat, prop_cls_core_feat - prop_cls_context_feat, prop_cls_context_feat), dim=-1)

        prop_iou_core_feat = self._aggregate_cap_features(cap_feat, cap_iou_occ)
        prop_iou_context_feat = self._aggregate_cap_features(cap_feat, 1 - cap_iou_occ)
        prop_iou_calib_feat = torch.cat((prop_iou_core_feat, prop_iou_core_feat - prop_iou_context_feat, prop_iou_context_feat), dim=-1)

        base_prob = F.softmax(prop_cas.transpose(-1, -2), dim=-1).detach()
        base_scalar_conf = torch.sigmoid(prop_attn.transpose(-1, -2)) * torch.sigmoid(prop_iou.transpose(-1, -2))
        raw_cls_gate = torch.sigmoid(self.cap_cls_gate(prop_cls_calib_feat))
        cls_gate_prior = ((1 - self.gate_prior_alpha) + self.gate_prior_alpha * (base_scalar_conf * base_prob)).clamp(min=0.0, max=1.0)
        prop_cls_gate = raw_cls_gate * cls_gate_prior
        raw_iou_gate = torch.sigmoid(self.cap_iou_gate(prop_iou_calib_feat))
        iou_gate_prior = ((1 - self.gate_prior_alpha) + self.gate_prior_alpha * (base_scalar_conf * base_prob[..., :-1].max(dim=-1, keepdim=True)[0])).clamp(min=0.0, max=1.0)
        prop_iou_gate = raw_iou_gate * iou_gate_prior
        prop_delta_cls = torch.tanh(self.cap_delta_cls(prop_cls_calib_feat))
        prop_delta_iou = torch.tanh(self.cap_delta_iou(prop_iou_calib_feat))

        prop_cap_iou = prop_iou_gate * prop_delta_iou
        core_prob = F.softmax(prop_core_cas, dim=-1)[..., :-1]
        context_prob = F.softmax(prop_context_cas, dim=-1)[..., :-1]
        cap_iou_support = cap_iou_occ.mean(dim=2)
        branch_quality = (core_prob - self.rescue_context_alpha * context_prob + self.rescue_iou_alpha * cap_iou_support).clamp(min=0.0)
        base_conf_cls = (base_scalar_conf * base_prob[..., :-1]).clamp(min=0.0, max=1.0)
        rescue_disagreement = F.relu(branch_quality - base_conf_cls)
        rescue_ambiguity = (4.0 * base_conf_cls * (1.0 - base_conf_cls)).clamp(min=0.0, max=1.0)
        rescue_ambiguity = (1 - self.rescue_ambiguity_alpha) + self.rescue_ambiguity_alpha * rescue_ambiguity
        prop_rescue_gate = rescue_disagreement * rescue_ambiguity
        effective_cls_gate = torch.clamp(prop_cls_gate[..., :-1] + prop_rescue_gate, max=1.5)
        prop_rescued_action_cas = prop_core_cas[..., :-1] + effective_cls_gate * prop_delta_cls[..., :-1]
        prop_rescued_bg_cas = prop_core_cas[..., -1:] + prop_cls_gate[..., -1:] * prop_delta_cls[..., -1:]
        prop_rescued_cas = torch.cat((prop_rescued_action_cas, prop_rescued_bg_cas), dim=-1)
        prop_repr = feat_fuse.transpose(-1, -2)
        return prop_cas, prop_attn, prop_iou, prop_rescued_cas, prop_dom_logit, prop_cap_iou, branch_quality, base_conf_cls, prop_repr


class P_MIL(torch.nn.Module):
    """
    PyTorch module for the Proposal-based Multiple Instance Learning (P-MIL) framework
    """
    def __init__(self, args):
        super().__init__()
        n_class = args.num_class
        dropout_ratio = args.dropout_ratio
        self.feat_dim = args.feature_size
        self.max_proposal = args.max_proposal
        self.roi_size = args.roi_size
        self.edge_size = max(1, self.roi_size // 6)
        self.mass_refine_alpha = args.mass_refine_alpha
        self.cap_iou_alpha = args.cap_iou_alpha
        self.cap_gate_prior_alpha = args.cap_gate_prior_alpha
        self.cap_rescue_context_alpha = args.cap_rescue_context_alpha
        self.cap_rescue_iou_alpha = args.cap_rescue_iou_alpha
        self.cap_rescue_ambiguity_alpha = args.cap_rescue_ambiguity_alpha
        self.cpm_residual_alpha = args.cpm_residual_alpha
        self.prop_v_backbone = Backbone_Proposal(
            self.feat_dim // 2,
            n_class,
            dropout_ratio,
            self.roi_size,
            gate_prior_alpha=self.cap_gate_prior_alpha,
            rescue_context_alpha=self.cap_rescue_context_alpha,
            rescue_iou_alpha=self.cap_rescue_iou_alpha,
            rescue_ambiguity_alpha=self.cap_rescue_ambiguity_alpha,
        )
        self.prop_f_backbone = Backbone_Proposal(
            self.feat_dim // 2,
            n_class,
            dropout_ratio,
            self.roi_size,
            gate_prior_alpha=self.cap_gate_prior_alpha,
            rescue_context_alpha=self.cap_rescue_context_alpha,
            rescue_iou_alpha=self.cap_rescue_iou_alpha,
            rescue_ambiguity_alpha=self.cap_rescue_ambiguity_alpha,
        )
        self.cpm = CrossProposalMemory(
            input_dim=self.feat_dim,
            n_class=n_class,
            dropout_ratio=dropout_ratio,
            embed_dim=args.cpm_embed_dim,
            bank_size=args.cpm_bank_size,
            topk=args.cpm_topk,
            temperature=args.cpm_temperature,
        )

    @torch.no_grad()
    def reset_cpm_memory(self):
        self.cpm.reset_memory()

    @torch.no_grad()
    def update_cpm_teacher(self, momentum):
        self.cpm.update_target_encoder(momentum)

    @torch.no_grad()
    def update_cpm_memory(self, outputs, labels, epoch, args):
        source_cas = 0.5 * (outputs['prop_v_refined_cas'] + outputs['prop_f_refined_cas'])
        self.cpm.update_memory(
            outputs['prop_repr'],
            outputs['prop_segments'],
            outputs['feature_lengths'],
            outputs['prop_span'],
            outputs['prop_scalar_conf'],
            outputs['prop_action_prob'],
            outputs['prop_mask'],
            source_cas,
            labels,
            outputs['video_indices'],
            args,
        )

    def extract_roi_features(self, features, proposals, is_training):
        """
        Extract region of interest (RoI) features from raw i3d features based on given proposals

        Inputs:
            features: list of [T, D] tensors
            proposals: list of [M, 2] tensors
            is_training: bool

        Outputs:
            prop_features:tensor of size [B, M, roi_size, D]
            prop_mask: tensor of size [B, M]
            prop_segments: tensor of size [B, M, 2]
            feature_lengths: tensor of size [B]
        """
        num_prop = [prop.shape[0] for prop in proposals]
        batch, max_num = len(proposals), max(num_prop)
        # Limit the max number of proposals during training
        if is_training:
            max_num = min(max_num, self.max_proposal)
        prop_features = torch.zeros((batch, max_num, self.roi_size, self.feat_dim)).to(features[0].device)
        prop_mask = torch.zeros((batch, max_num)).to(features[0].device)
        prop_segments = torch.zeros((batch, max_num, 2)).to(features[0].device)
        feature_lengths = torch.tensor([feature.shape[0] for feature in features], dtype=torch.float32, device=features[0].device)

        for i in range(batch):
            feature = features[i]
            proposal = proposals[i]
            if num_prop[i] > max_num:
                sampled_idx = torch.randperm(num_prop[i], device=proposal.device)[:max_num]
                proposal = proposal[sampled_idx]

            # Extend the proposal by 25% of its length at both sides
            start, end = proposal[:, 0], proposal[:, 1]
            len_prop = end - start
            start_ext = start - 0.25 * len_prop
            end_ext = end + 0.25 * len_prop
            # Fill in blank at edge of the feature, offset 0.5, for more accurate RoI_Align results
            fill_len = torch.ceil(0.25 * len_prop.max()).long() + 1                         # +1 because of offset 0.5
            fill_blank = torch.zeros(fill_len, self.feat_dim).to(feature.device)
            feature = torch.cat([fill_blank, feature, fill_blank], dim=0)
            start_ext = start_ext + fill_len - 0.5
            end_ext = end_ext + fill_len - 0.5
            proposal_ext = torch.stack((start_ext, end_ext), dim=1)
            
            # Extract RoI features using RoI Align operation
            y1, y2 = proposal_ext[:, 0], proposal_ext[:, 1]
            x1, x2 = torch.zeros_like(y1), torch.ones_like(y2)
            boxes = torch.stack((x1, y1, x2, y2), dim=1)                                    # [M, 4]
            feature = feature.transpose(0, 1).unsqueeze(0).unsqueeze(3)                     # [1, D, T, 1]
            feat_roi = torchvision.ops.roi_align(feature, [boxes], [self.roi_size, 1])      # [M, D, roi_size, 1]
            feat_roi = feat_roi.squeeze(3).transpose(1, 2)                                  # [M, roi_size, D]
            prop_features[i, :proposal.shape[0], :, :] = feat_roi                           # [B, M, roi_size, D]
            prop_mask[i, :proposal.shape[0]] = 1                                            # [B, M]
            prop_segments[i, :proposal.shape[0], :] = proposal                              # [B, M, 2]

        return prop_features, prop_mask, prop_segments, feature_lengths

    def forward(self, features, proposals, is_training=True, video_indices=None):
        """
        Inputs:
            features: list of [T, D] tensors
            proposals: list of [M, 2] tensors
            is_training: bool

        Outputs:
            outputs: dictionary
        """
        prop_features, prop_mask, prop_segments, feature_lengths = self.extract_roi_features(features, proposals, is_training)
        prop_v_features = prop_features[..., :self.feat_dim // 2]
        prop_f_features = prop_features[..., self.feat_dim // 2:]

        prop_v_cas, prop_v_attn, prop_v_iou, prop_v_rescued_cas, prop_v_dom_logit, prop_v_cap_iou, prop_v_rescue_quality, prop_v_base_conf, prop_v_repr = self.prop_v_backbone(prop_v_features)
        prop_f_cas, prop_f_attn, prop_f_iou, prop_f_rescued_cas, prop_f_dom_logit, prop_f_cap_iou, prop_f_rescue_quality, prop_f_base_conf, prop_f_repr = self.prop_f_backbone(prop_f_features)
        prop_v_cas = prop_v_cas.transpose(-1, -2)                       # [B, M, C]
        prop_f_cas = prop_f_cas.transpose(-1, -2)                       # [B, M, C]
        prop_v_attn = prop_v_attn.transpose(-1, -2)                     # [B, M, 1]
        prop_f_attn = prop_f_attn.transpose(-1, -2)                     # [B, M, 1]
        prop_v_iou = prop_v_iou.transpose(-1, -2)                       # [B, M, 1]
        prop_f_iou = prop_f_iou.transpose(-1, -2)                       # [B, M, 1]
        prop_v_refined_iou = prop_v_iou + self.cap_iou_alpha * prop_v_cap_iou
        prop_f_refined_iou = prop_f_iou + self.cap_iou_alpha * prop_f_cap_iou
        prop_v_refined_cas = self.compose_refined_cas(prop_v_cas, prop_v_rescued_cas, detach_base=False)
        prop_f_refined_cas = self.compose_refined_cas(prop_f_cas, prop_f_rescued_cas, detach_base=False)
        avg_attn = 0.5 * (torch.sigmoid(prop_v_attn) + torch.sigmoid(prop_f_attn))
        avg_iou = 0.5 * (torch.sigmoid(prop_v_refined_iou) + torch.sigmoid(prop_f_refined_iou))
        avg_prob = F.softmax(0.5 * (prop_v_refined_cas + prop_f_refined_cas), dim=-1)
        prop_scalar_conf = avg_attn * avg_iou
        prop_action_prob = avg_prob[..., :-1].max(dim=-1, keepdim=True)[0]
        prop_len = (prop_segments[..., 1] - prop_segments[..., 0]).clamp(min=1e-4)
        prop_span = torch.log((prop_len / feature_lengths.unsqueeze(1)).clamp(min=1e-4)).unsqueeze(-1)
        prop_repr = torch.cat((prop_v_repr, prop_f_repr), dim=-1)
        if video_indices is None:
            video_indices = torch.full((prop_features.shape[0],), -1, dtype=torch.long, device=prop_features.device)
        cpm_output = self.cpm(
            prop_repr,
            prop_segments,
            feature_lengths,
            prop_span.detach(),
            prop_scalar_conf.detach(),
            prop_action_prob.detach(),
            prop_mask,
            query_video_ids=video_indices,
        )
        prop_cpm_cls_delta = cpm_output['cpm_cls_delta']
        prop_cpm_bonus = cpm_output['cpm_bonus']
        prop_cpm_cls_residual = prop_cpm_cls_delta + 0.35 * (2.0 * prop_cpm_bonus - 1.0)
        prop_cpm_delta_full = torch.cat((prop_cpm_cls_residual, torch.zeros_like(prop_cpm_cls_residual[..., :1])), dim=-1)
        prop_v_cpm_cas = prop_v_refined_cas + self.cpm_residual_alpha * prop_cpm_delta_full
        prop_f_cpm_cas = prop_f_refined_cas + self.cpm_residual_alpha * prop_cpm_delta_full

        outputs = {
            'prop_v_cas': prop_v_cas,                       # [B, M, C]
            'prop_f_cas': prop_f_cas,                       # [B, M, C]
            'prop_v_attn': prop_v_attn,                     # [B, M, 1]
            'prop_f_attn': prop_f_attn,                     # [B, M, 1]
            'prop_v_iou': prop_v_iou,                       # [B, M, 1]
            'prop_f_iou': prop_f_iou,                       # [B, M, 1]
            'prop_v_rescued_cas': prop_v_rescued_cas,       # [B, M, C]
            'prop_f_rescued_cas': prop_f_rescued_cas,       # [B, M, C]
            'prop_v_dom_logit': prop_v_dom_logit,           # [B, M, C]
            'prop_f_dom_logit': prop_f_dom_logit,           # [B, M, C]
            'prop_v_cap_iou': prop_v_cap_iou,               # [B, M, 1]
            'prop_f_cap_iou': prop_f_cap_iou,               # [B, M, 1]
            'prop_v_rescue_quality': prop_v_rescue_quality, # [B, M, C-1]
            'prop_f_rescue_quality': prop_f_rescue_quality, # [B, M, C-1]
            'prop_v_base_conf': prop_v_base_conf,           # [B, M, C-1]
            'prop_f_base_conf': prop_f_base_conf,           # [B, M, C-1]
            'prop_repr': prop_repr,                         # [B, M, 2D]
            'prop_span': prop_span,                         # [B, M, 1]
            'prop_scalar_conf': prop_scalar_conf,           # [B, M, 1]
            'prop_action_prob': prop_action_prob,           # [B, M, 1]
            'prop_cpm_positive': cpm_output['positive_response'],   # [B, M, C-1]
            'prop_cpm_bonus': prop_cpm_bonus,                       # [B, M, C-1]
            'prop_cpm_pos_valid': cpm_output['cpm_pos_valid'],      # [B, M, C-1]
            'prop_v_refined_cas': prop_v_refined_cas,       # [B, M, C]
            'prop_f_refined_cas': prop_f_refined_cas,       # [B, M, C]
            'prop_v_cpm_cas': prop_v_cpm_cas,               # [B, M, C]
            'prop_f_cpm_cas': prop_f_cpm_cas,               # [B, M, C]
            'prop_v_refined_iou': prop_v_refined_iou,       # [B, M, 1]
            'prop_f_refined_iou': prop_f_refined_iou,       # [B, M, 1]
            'prop_mask': prop_mask,                         # [B, M]
            'prop_segments': prop_segments,                 # [B, M, 2]
            'feature_lengths': feature_lengths,             # [B]
            'video_indices': video_indices,                 # [B]
        }
        return outputs

    def get_consistency_weight(self, current, rampup_length):
        """
        Exponential rampup from https://arxiv.org/abs/1610.02242
        """
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    def compose_refined_cas(self, base_cas, rescued_cas, detach_base=False):
        base = base_cas.detach() if detach_base else base_cas
        return (1 - self.mass_refine_alpha) * base + self.mass_refine_alpha * rescued_cas

    def segments_iou(self, segments1, segments2):
        """
        Inputs:
            segments1: tensor of size [M1, 2]
            segments2: tensor of size [M2, 2]

        Outputs:
            iou_temp: tensor of size [M1, M2]
        """
        segments1 = segments1.unsqueeze(1)                          # [M1, 1, 2]
        segments2 = segments2.unsqueeze(0)                          # [1, M2, 2]
        tt1 = torch.maximum(segments1[..., 0], segments2[..., 0])   # [M1, M2]
        tt2 = torch.minimum(segments1[..., 1], segments2[..., 1])   # [M1, M2]
        intersection = tt2 - tt1
        union = (segments1[..., 1] - segments1[..., 0]) + (segments2[..., 1] - segments2[..., 0]) - intersection
        iou = intersection / (union + 1e-6)                         # [M1, M2]
        # Remove negative values
        iou_temp = torch.zeros_like(iou)
        iou_temp[iou > 0] = iou[iou > 0]
        return iou_temp

    def criterion(self, outputs, labels, proposals, epoch, args):
        """
        Compute the total loss function

        Inputs: 
            outputs: dictionary
            labels: tensor of size [B, C]
            proposals: list of [M, 2] tensors
            epoch: int
            args: argparse.Namespace

        Outputs:
            loss_dict: dictionary
        """
        prop_v_cas, prop_v_attn, prop_v_iou_logit = outputs['prop_v_cas'], outputs['prop_v_attn'], outputs['prop_v_iou']
        prop_f_cas, prop_f_attn, prop_f_iou_logit = outputs['prop_f_cas'], outputs['prop_f_attn'], outputs['prop_f_iou']
        prop_v_rescued_cas, prop_f_rescued_cas = outputs['prop_v_rescued_cas'], outputs['prop_f_rescued_cas']
        prop_v_dom_logit, prop_f_dom_logit = outputs['prop_v_dom_logit'], outputs['prop_f_dom_logit']
        prop_v_cap_iou, prop_f_cap_iou = outputs['prop_v_cap_iou'], outputs['prop_f_cap_iou']
        prop_v_rescue_quality, prop_f_rescue_quality = outputs['prop_v_rescue_quality'], outputs['prop_f_rescue_quality']
        prop_v_base_conf, prop_f_base_conf = outputs['prop_v_base_conf'], outputs['prop_f_base_conf']
        prop_v_cpm_cas, prop_f_cpm_cas = outputs['prop_v_cpm_cas'], outputs['prop_f_cpm_cas']
        prop_cpm_positive = outputs['prop_cpm_positive']
        prop_cpm_bonus = outputs['prop_cpm_bonus']
        prop_cpm_pos_valid = outputs['prop_cpm_pos_valid']
        prop_mask_float = outputs['prop_mask']
        prop_segments = outputs['prop_segments']
        prop_mask = prop_mask_float

        prop_v_attn = torch.sigmoid(prop_v_attn)                        # [B, M, 1]
        prop_f_attn = torch.sigmoid(prop_f_attn)                        # [B, M, 1]
        prop_v_iou = torch.sigmoid(prop_v_iou_logit)                    # [B, M, 1]
        prop_f_iou = torch.sigmoid(prop_f_iou_logit)                    # [B, M, 1]
        prop_v_cap_iou = torch.sigmoid(prop_v_iou_logit.detach() + self.cap_iou_alpha * prop_v_cap_iou)
        prop_f_cap_iou = torch.sigmoid(prop_f_iou_logit.detach() + self.cap_iou_alpha * prop_f_cap_iou)
        prop_v_refined_branch = self.compose_refined_cas(prop_v_cas, prop_v_rescued_cas, detach_base=True)
        prop_f_refined_branch = self.compose_refined_cas(prop_f_cas, prop_f_rescued_cas, detach_base=True)
        aligned_proposals = [prop_segments[b][prop_mask_float[b].bool()] for b in range(prop_segments.shape[0])]
        prop_mask = prop_mask.unsqueeze(2).bool()                       # [B, M, 1]
        prop_mask_cas = prop_mask.repeat((1, 1, prop_v_cas.shape[2]))   # [B, M, C]

        # proposal classification loss
        prop_v_cas_supp = prop_v_cas * prop_v_attn
        prop_f_cas_supp = prop_f_cas * prop_f_attn
        loss_prop_mil_orig_v = self.prop_topk_loss(prop_v_cas,      labels, prop_mask_cas, is_back=True,  topk=args.k)
        loss_prop_mil_orig_f = self.prop_topk_loss(prop_f_cas,      labels, prop_mask_cas, is_back=True,  topk=args.k)
        loss_prop_mil_supp_v = self.prop_topk_loss(prop_v_cas_supp, labels, prop_mask_cas, is_back=False, topk=args.k)
        loss_prop_mil_supp_f = self.prop_topk_loss(prop_f_cas_supp, labels, prop_mask_cas, is_back=False, topk=args.k)
        loss_prop_mil_refined_v = self.prop_topk_loss(prop_v_refined_branch, labels, prop_mask_cas, is_back=False, topk=args.k)
        loss_prop_mil_refined_f = self.prop_topk_loss(prop_f_refined_branch, labels, prop_mask_cas, is_back=False, topk=args.k)
        loss_prop_mil_cpm_v = self.prop_topk_loss(prop_v_cpm_cas, labels, prop_mask_cas, is_back=False, topk=args.k)
        loss_prop_mil_cpm_f = self.prop_topk_loss(prop_f_cpm_cas, labels, prop_mask_cas, is_back=False, topk=args.k)

        # Instance-level Rank Consistency (IRC) loss
        loss_prop_irc_v = self.prop_irc_loss(prop_v_cas, prop_f_cas, prop_f_attn, labels, prop_mask, prop_mask_cas, aligned_proposals)
        loss_prop_irc_f = self.prop_irc_loss(prop_f_cas, prop_v_cas, prop_v_attn, labels, prop_mask, prop_mask_cas, aligned_proposals)

        # proposal completeness loss
        loss_prop_comp_v = self.prop_comp_loss(prop_v_iou, prop_f_attn, prop_mask, aligned_proposals, args.gamma)
        loss_prop_comp_f = self.prop_comp_loss(prop_f_iou, prop_v_attn, prop_mask, aligned_proposals, args.gamma)
        loss_prop_cap_comp_v = self.prop_comp_loss(prop_v_cap_iou, prop_f_attn.detach(), prop_mask, aligned_proposals, args.gamma)
        loss_prop_cap_comp_f = self.prop_comp_loss(prop_f_cap_iou, prop_v_attn.detach(), prop_mask, aligned_proposals, args.gamma)
        if args.weight_loss_cap_rescue > 0:
            loss_cap_rescue = self.cap_positive_rescue_loss(
                prop_v_rescue_quality,
                prop_f_rescue_quality,
                prop_v_base_conf,
                prop_f_base_conf,
                prop_v_rescued_cas,
                prop_f_rescued_cas,
                prop_v_attn,
                prop_f_attn,
                prop_v_iou,
                prop_f_iou,
                labels,
                prop_mask,
                args,
            )
        else:
            loss_cap_rescue = prop_v_cas.new_zeros(())
        loss_cap_action_dominance = self.cap_action_dominance_loss(prop_v_dom_logit, prop_f_dom_logit, outputs['prop_v_refined_cas'], outputs['prop_f_refined_cas'], prop_v_attn, prop_f_attn, prop_v_iou, prop_f_iou, labels, prop_mask, args)
        cpm_sup_v_cas = outputs['prop_v_refined_cas'].detach()
        cpm_sup_f_cas = outputs['prop_f_refined_cas'].detach()
        cpm_sup_v_iou = outputs['prop_v_refined_iou'].detach()
        cpm_sup_f_iou = outputs['prop_f_refined_iou'].detach()
        loss_cpm_cls = self.cpm_cls_loss(
            prop_cpm_bonus,
            prop_cpm_pos_valid,
            cpm_sup_v_cas,
            cpm_sup_f_cas,
            prop_v_attn,
            prop_f_attn,
            cpm_sup_v_iou,
            cpm_sup_f_iou,
            labels,
            prop_mask,
            args,
        )
        loss_cpm_rank = self.cpm_rank_loss(
            prop_cpm_bonus,
            prop_cpm_positive,
            prop_cpm_pos_valid,
            cpm_sup_v_cas,
            cpm_sup_f_cas,
            prop_v_attn,
            prop_f_attn,
            cpm_sup_v_iou,
            cpm_sup_f_iou,
            labels,
            prop_mask,
            args,
        )

        loss_prop_mil_orig = args.weight_loss_prop_mil_orig * (loss_prop_mil_orig_v + loss_prop_mil_orig_f) / 2
        loss_prop_mil_supp = args.weight_loss_prop_mil_supp * (loss_prop_mil_supp_v + loss_prop_mil_supp_f) / 2
        loss_prop_mil_refined = args.weight_loss_prop_mil_refined * (loss_prop_mil_refined_v + loss_prop_mil_refined_f) / 2
        if epoch > args.cpm_warmup_epoch:
            loss_mil_cpm = args.weight_loss_mil_cpm * (loss_prop_mil_cpm_v + loss_prop_mil_cpm_f) / 2
        else:
            loss_mil_cpm = (prop_v_cpm_cas.sum() + prop_f_cpm_cas.sum()) * 0
        loss_prop_irc = args.weight_loss_prop_irc * (loss_prop_irc_v + loss_prop_irc_f) / 2 * self.get_consistency_weight(epoch, args.rampup_length)
        loss_prop_comp = args.weight_loss_prop_comp * (loss_prop_comp_v + loss_prop_comp_f) / 2 * self.get_consistency_weight(epoch, args.rampup_length)
        loss_cap_comp = args.weight_loss_cap_comp * (loss_prop_cap_comp_v + loss_prop_cap_comp_f) / 2 * self.get_consistency_weight(epoch, args.rampup_length)
        loss_cap_rescue = args.weight_loss_cap_rescue * loss_cap_rescue * self.get_consistency_weight(epoch, args.rampup_length)
        loss_cap_action_dominance = args.weight_loss_cap_action_dominance * loss_cap_action_dominance * self.get_consistency_weight(epoch, args.rampup_length)
        loss_cpm_cls = args.weight_loss_cpm_cls * loss_cpm_cls * self.get_consistency_weight(epoch, args.rampup_length)
        loss_cpm_rank = args.weight_loss_cpm_rank * loss_cpm_rank * self.get_consistency_weight(epoch, args.rampup_length)
        loss_total = loss_prop_mil_orig + loss_prop_mil_supp + loss_prop_mil_refined + loss_prop_irc + loss_prop_comp + loss_cap_comp + loss_cap_rescue + loss_cap_action_dominance + loss_cpm_cls + loss_cpm_rank
        loss_total = loss_total + loss_mil_cpm

        loss_dict = {
            'loss_total': loss_total,
            'loss_prop_mil_orig': loss_prop_mil_orig,
            'loss_prop_mil_supp': loss_prop_mil_supp,
            'loss_prop_mil_refined': loss_prop_mil_refined,
            'loss_mil_cpm': loss_mil_cpm,
            'loss_prop_irc': loss_prop_irc,
            'loss_prop_comp': loss_prop_comp,
            'loss_cap_comp': loss_cap_comp,
            'loss_cap_rescue': loss_cap_rescue,
            'loss_cap_action_dominance': loss_cap_action_dominance,
            'loss_cpm_cls': loss_cpm_cls,
            'loss_cpm_rank': loss_cpm_rank,
        }
        return loss_dict

    def prop_topk_loss(self, cas, labels, mask_cas, is_back=True, topk=8):
        """
        Compute the topk classification loss

        Inputs:
            cas: tensor of size [B, M, C]
            labels: tensor of size [B, C]
            mask_cas: tensor of size [B, M, C]
            is_back: bool
            topk: int

        Outputs:
            loss_mil: tensor
        """
        if is_back:
            labels_with_back = torch.cat((labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat((labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        labels_with_back = labels_with_back / (torch.sum(labels_with_back, dim=-1, keepdim=True) + 1e-4)

        loss_mil = 0
        for b in range(cas.shape[0]):
            cas_b = cas[b][mask_cas[b]].reshape((-1, cas.shape[-1]))
            topk_val, _ = torch.topk(cas_b, k=max(1, int(cas_b.shape[-2] // topk)), dim=-2)
            video_score = torch.mean(topk_val, dim=-2)
            loss_mil += - (labels_with_back[b] * F.log_softmax(video_score, dim=-1)).sum(dim=-1).mean()
        loss_mil /= cas.shape[0]

        return loss_mil

    def prop_irc_loss(self, cas_stu, cas_tea, attn, labels, mask, mask_cas, proposals):
        """
        Compute the Instance-level Rank Consistency (IRC) loss

        Inputs:
            cas_stu: tensor of size [B, M, C]
            cas_tea: tensor of size [B, M, C]
            attn: tensor of size [B, M, 1]
            labels: tensor of size [B, C]
            mask: bool tensor of size [B, M, 1]
            mask_cas: bool tensor of size [B, M, C]
            proposals: list of [M, 2] tensors

        Outputs:
            loss_irc: tensor
        """
        loss_irc = 0
        for b in range(len(proposals)):
            attn_b = attn[b][mask[b]]
            cas_stu_b = cas_stu[b][mask_cas[b]].reshape((-1, mask_cas.shape[-1]))
            cas_tea_b = cas_tea[b][mask_cas[b]].reshape((-1, mask_cas.shape[-1]))
            proposals_iou = self.segments_iou(proposals[b], proposals[b])
            # used to mask out non-overlapping proposals
            proposals_mask = torch.zeros_like(proposals_iou)
            proposals_mask[proposals_iou <= 0] = -1e3
            proposals_mask[proposals_iou > 0] = 0

            loss_irc_b = 0
            for c in torch.where(labels[b])[0]:
                score_stu = cas_stu_b[:, c]
                score_tea = cas_tea_b[:, c]

                # the KL loss is only computed for proposals that overlap with the given proposal
                softmax_tea = F.softmax(proposals_mask + score_tea.unsqueeze(0), dim=1)
                softmax_stu = F.log_softmax(proposals_mask + score_stu.unsqueeze(0), dim=1)
                loss_kl_matrix = F.kl_div(softmax_stu, softmax_tea.detach(), reduction='none').sum(-1)

                # eliminate the low-confidence proposals
                retained = attn_b > torch.mean(attn_b)
                loss_irc_b += loss_kl_matrix[retained].mean()
            loss_irc_b /= labels[b].sum()
            loss_irc += loss_irc_b
        loss_irc /= len(proposals)

        return loss_irc

    def prop_comp_loss(self, pred_iou, attn, mask, proposals, gamma):
        """
        Compute the completeness loss

        Inputs:
            pred_iou: tensor of size [B, M, 1]
            attn: tensor of size [B, M, 1]
            mask: bool tensor of size [B, M, 1]
            proposals: list of [M, 2] tensors
            gamma: float

        Outputs:
            loss_comp: tensor
        """
        loss_comp = 0
        for b in range(len(proposals)):
            attn_b = attn[b][mask[b]]
            pred_iou_b = pred_iou[b][mask[b]]
            proposals_iou = self.segments_iou(proposals[b], proposals[b])
            proposals_mask = proposals_iou > 0

            # using NMS to select the pseudo instances, the running speed is slow
            choiced = []
            retained = attn_b > gamma * torch.max(attn_b)
            while retained.sum() > 0:
                max_idx = torch.max(attn_b[retained], dim=0)[1]
                max_idx = torch.where(retained)[0][max_idx]
                overlap = proposals_mask[max_idx]
                retained[overlap] = False
                choiced.append(max_idx)
            choiced = torch.stack(choiced, dim=0)
            pseudo_instances = proposals[b][choiced]

            pseudo_iou = self.segments_iou(proposals[b], pseudo_instances)
            pseudo_iou = torch.max(pseudo_iou, dim=1)[0]
            loss_comp += F.mse_loss(pred_iou_b, pseudo_iou)
        loss_comp /= len(proposals)

        return loss_comp

    def cap_positive_rescue_loss(self, rescue_quality_v, rescue_quality_f, base_conf_v, base_conf_f, rescued_cas_v, rescued_cas_f, attn_v, attn_f, iou_v, iou_f, labels, mask, args):
        """
        Encourage CAP to rescue true positives that show strong action-core
        quality but are still under-scored by the base confidence.
        """
        num_action_class = labels.shape[1]
        rescue_quality = 0.5 * (rescue_quality_v + rescue_quality_f)
        base_conf = 0.5 * (base_conf_v + base_conf_f)
        rescued_prob = F.softmax(0.5 * (rescued_cas_v + rescued_cas_f), dim=-1)[:, :, :num_action_class]
        conf = (0.5 * (attn_v + attn_f) * 0.5 * (iou_v + iou_f)).detach().squeeze(-1)
        valid_mask = mask.squeeze(-1)

        loss_rescue = rescue_quality.new_zeros(())
        count = 0
        for b in range(rescue_quality.shape[0]):
            valid = valid_mask[b]
            if valid.sum() < 4:
                continue

            positive_cls = torch.where(labels[b] > 0)[0]
            if positive_cls.numel() == 0:
                continue

            quality_b = rescue_quality[b][valid]
            base_conf_b = base_conf[b][valid]
            rescued_prob_b = rescued_prob[b][valid]
            conf_b = conf[b][valid]
            for c in positive_cls:
                disagreement = quality_b[:, c] - base_conf_b[:, c]
                rescue_mask = disagreement > 0
                if rescue_mask.sum() == 0:
                    continue

                rescue_score = disagreement.clamp(min=0) * quality_b[:, c]
                rescue_score = rescue_score.masked_fill(~rescue_mask, -1)
                pos_k = min(args.cap_rescue_topk, int(rescue_mask.sum().item()))
                if pos_k < 1:
                    continue
                pos_idx = torch.topk(rescue_score, k=pos_k, dim=0).indices

                all_idx = torch.arange(rescue_mask.shape[0], device=rescue_mask.device)
                neg_mask = (all_idx.unsqueeze(1) != pos_idx.reshape(1, -1)).all(dim=1)
                if neg_mask.sum() == 0:
                    continue

                neg_score = (base_conf_b[:, c] - quality_b[:, c]).clamp(min=0) * (0.5 * base_conf_b[:, c] + 0.5 * conf_b)
                neg_score = neg_score.masked_fill(~neg_mask, -1)
                neg_valid = neg_score > 0
                if neg_valid.sum() == 0:
                    continue
                neg_k = min(args.cap_rescue_topk, int(neg_valid.sum().item()))
                neg_idx = torch.topk(neg_score, k=neg_k, dim=0).indices

                pos_pred = rescued_prob_b[pos_idx, c]
                neg_pred = rescued_prob_b[neg_idx, c]
                pair_loss = F.relu(args.cap_rescue_margin - pos_pred.unsqueeze(1) + neg_pred.unsqueeze(0))
                pair_weight = torch.sqrt(
                    (rescue_score[pos_idx].clamp(min=1e-6).unsqueeze(1) * neg_score[neg_idx].clamp(min=1e-6).unsqueeze(0))
                )
                loss_rescue = loss_rescue + (pair_loss * pair_weight).sum() / (pair_weight.sum() + 1e-6)
                count += 1

        if count == 0:
            return rescue_quality.new_zeros(())
        return loss_rescue / count

    def cap_action_dominance_loss(self, prop_v_dom_logit, prop_f_dom_logit, prop_v_cas, prop_f_cas, prop_v_attn, prop_f_attn, prop_v_iou, prop_f_iou, labels, mask, args):
        """
        Top positive proposals should show stronger inner-than-outer class evidence.
        """
        num_action_class = labels.shape[1]
        dom_logit = 0.5 * (prop_v_dom_logit + prop_f_dom_logit)[:, :, :num_action_class]        # [B, M, C]
        cls_prob = F.softmax(0.5 * (prop_v_cas + prop_f_cas), dim=-1).detach()[:, :, :num_action_class]
        conf = (0.5 * (prop_v_attn + prop_f_attn) * 0.5 * (prop_v_iou + prop_f_iou)).detach().squeeze(-1)
        valid_mask = mask.squeeze(-1)

        loss_dom = dom_logit.new_zeros(())
        count = 0
        for b in range(dom_logit.shape[0]):
            valid = valid_mask[b]
            if valid.sum() < 4:
                continue

            positive_cls = torch.where(labels[b] > 0)[0]
            if positive_cls.numel() == 0:
                continue

            dom_b = dom_logit[b][valid]
            conf_b = conf[b][valid]
            cls_prob_b = cls_prob[b][valid]
            for c in positive_cls:
                pseudo_conf = conf_b * cls_prob_b[:, c]
                if pseudo_conf.numel() == 0:
                    continue
                k = max(1, pseudo_conf.shape[0] // max(1, args.dominance_topk))
                top_idx = torch.topk(pseudo_conf, k=k, dim=0).indices
                dom_c = dom_b[top_idx, c]
                weight = pseudo_conf[top_idx].detach()
                loss_c = F.relu(args.dominance_margin - dom_c)
                loss_dom = loss_dom + (loss_c * weight).sum() / (weight.sum() + 1e-6)
                count += 1

        if count == 0:
            return dom_logit.new_zeros(())
        return loss_dom / count

    def cpm_cls_loss(self, cpm_bonus, cpm_valid, select_cas_v, select_cas_f, attn_v, attn_f, select_iou_v, select_iou_f, labels, mask, args):
        """
        Train Proposal-level Cross-video Support Memory as a positive-support
        re-ranker. Present classes should receive higher support bonus on
        pseudo-positive anchors, while absent classes should be suppressed on the
        same anchors whenever they have valid support memory.
        """
        cls_prob = F.softmax(0.5 * (select_cas_v + select_cas_f), dim=-1).detach()[:, :, :labels.shape[1]]
        conf = (0.5 * (attn_v + attn_f) * 0.5 * (torch.sigmoid(select_iou_v) + torch.sigmoid(select_iou_f))).detach().squeeze(-1)
        valid_mask = mask.squeeze(-1).bool()

        loss_cls = cpm_bonus.sum() * 0
        count = 0
        for b in range(cpm_bonus.shape[0]):
            valid = valid_mask[b]
            if valid.sum() == 0:
                continue

            positive_cls = torch.where(labels[b] > 0)[0]
            negative_cls = torch.where(labels[b] == 0)[0]
            if positive_cls.numel() == 0:
                continue

            bonus_b = cpm_bonus[b][valid]
            valid_b = cpm_valid[b][valid] > 0
            cls_prob_b = cls_prob[b][valid]
            conf_b = conf[b][valid]
            for c in positive_cls.tolist():
                pseudo_score = conf_b * cls_prob_b[:, c]
                if pseudo_score.numel() == 0 or torch.max(pseudo_score) <= 0:
                    continue

                k = max(1, pseudo_score.shape[0] // max(1, args.cpm_anchor_topk_div))
                anchor_idx = torch.topk(pseudo_score, k=k, dim=0).indices
                pos_anchor_mask = valid_b[anchor_idx, c]
                if pos_anchor_mask.sum() == 0:
                    continue
                anchor_idx = anchor_idx[pos_anchor_mask]
                weight = pseudo_score[anchor_idx].detach()

                pos_bonus = bonus_b[anchor_idx, c].clamp(min=1e-6, max=1 - 1e-6)
                pos_loss = -(weight * torch.log(pos_bonus)).sum() / (weight.sum() + 1e-6)

                if negative_cls.numel() > 0:
                    neg_bonus = bonus_b[anchor_idx][:, negative_cls].clamp(min=1e-6, max=1 - 1e-6)
                    neg_valid = valid_b[anchor_idx][:, negative_cls]
                    if neg_valid.sum() > 0:
                        neg_bonus = neg_bonus[neg_valid]
                        neg_weight = weight.unsqueeze(1).expand(-1, negative_cls.numel())[neg_valid]
                        neg_loss = -(neg_weight * torch.log(1 - neg_bonus)).sum() / (neg_weight.sum() + 1e-6)
                    else:
                        neg_loss = pos_loss.new_zeros(())
                else:
                    neg_loss = pos_loss.new_zeros(())

                loss_cls = loss_cls + 0.5 * (pos_loss + neg_loss)
                count += 1

        if count == 0:
            return cpm_bonus.sum() * 0
        return loss_cls / count

    def cpm_rank_loss(self, cpm_bonus, cpm_align, cpm_valid, select_cas_v, select_cas_f, attn_v, attn_f, select_iou_v, select_iou_f, labels, mask, args):
        """
        Positive support for present classes should be more discriminative than
        support retrieved for absent classes on the same pseudo-positive
        anchors, both in support bonus and in prototype similarity.
        """
        cls_prob = F.softmax(0.5 * (select_cas_v + select_cas_f), dim=-1).detach()[:, :, :labels.shape[1]]
        conf = (0.5 * (attn_v + attn_f) * 0.5 * (torch.sigmoid(select_iou_v) + torch.sigmoid(select_iou_f))).detach().squeeze(-1)
        valid_mask = mask.squeeze(-1).bool()

        loss_rank = (cpm_bonus.sum() + cpm_align.sum()) * 0
        count = 0
        for b in range(cpm_bonus.shape[0]):
            valid = valid_mask[b]
            if valid.sum() == 0:
                continue
            positive_cls = torch.where(labels[b] > 0)[0]
            negative_cls = torch.where(labels[b] == 0)[0]
            if positive_cls.numel() == 0 or negative_cls.numel() == 0:
                continue

            bonus_b = cpm_bonus[b][valid]
            align_b = cpm_align[b][valid]
            valid_b = cpm_valid[b][valid] > 0
            cls_prob_b = cls_prob[b][valid]
            conf_b = conf[b][valid]
            for c in positive_cls.tolist():
                pseudo_score = conf_b * cls_prob_b[:, c]
                if pseudo_score.numel() == 0 or torch.max(pseudo_score) <= 0:
                    continue
                k = max(1, pseudo_score.shape[0] // max(1, args.cpm_anchor_topk_div))
                anchor_idx = torch.topk(pseudo_score, k=k, dim=0).indices
                pos_anchor_mask = valid_b[anchor_idx, c]
                if pos_anchor_mask.sum() == 0:
                    continue
                anchor_idx = anchor_idx[pos_anchor_mask]
                if anchor_idx.numel() == 0:
                    continue
                pos_bonus = bonus_b[anchor_idx, c]
                pos_align = align_b[anchor_idx, c]
                weight = pseudo_score[anchor_idx].detach()
                neg_valid = valid_b[anchor_idx][:, negative_cls]
                if neg_valid.sum() > 0:
                    neg_bonus_all = bonus_b[anchor_idx][:, negative_cls].masked_fill(~neg_valid, -1e4)
                    neg_align_all = align_b[anchor_idx][:, negative_cls].masked_fill(~neg_valid, -1e4)
                    neg_bonus = neg_bonus_all.max(dim=-1)[0]
                    neg_align = neg_align_all.max(dim=-1)[0]
                    rank_bonus = F.relu(args.cpm_rank_margin - pos_bonus + neg_bonus)
                    rank_align = F.relu(args.cpm_rank_margin - pos_align + neg_align)
                    rank_loss = 0.5 * (rank_bonus + rank_align)
                    loss_rank = loss_rank + (rank_loss * weight).sum() / (weight.sum() + 1e-6)
                    count += 1

        if count == 0:
            return (cpm_bonus.sum() + cpm_align.sum()) * 0
        return loss_rank / count
