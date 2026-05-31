import argparse


parser = argparse.ArgumentParser(description='CAP/CPM proposal-level WTAL')

# basic settings
parser.add_argument('--exp_dir', type=str, default='outputs', help='experiment output directory')
parser.add_argument('--run_type', type=str, default='train', choices=('train', 'test'), help='train or test')

# dataset parameters
parser.add_argument('--dataset_name', type=str, default='Thumos14reduced', help='dataset name')
parser.add_argument('--dataset_root', type=str, default='data/Thumos14reduced', help='dataset root path')
parser.add_argument('--proposal_dir', type=str, default='proposals', help='directory containing proposal json files')
parser.add_argument('--base_method', type=str, default='base', help='proposal source name')

# model parameters
parser.add_argument('--num_class', type=int, default=20, help='number of foreground classes')
parser.add_argument('--feature_size', type=int, default=2048, help='I3D feature dimension')
parser.add_argument('--roi_size', type=int, default=12, help='number of RoI bins per proposal')
parser.add_argument('--max_proposal', type=int, default=1000, help='maximum proposals per video during training')
parser.add_argument('--pretrained_ckpt', type=str, default=None, help='checkpoint used for initialization or testing')
parser.add_argument('--mass_refine_alpha', type=float, default=0.2, help='blend ratio for CAP-refined CAS')

parser.add_argument('--cap_iou_alpha', '--bin_iou_alpha', dest='cap_iou_alpha', type=float, default=0.3,
                    help='residual strength of CAP on proposal completeness logits')
parser.add_argument('--cap_gate_prior_alpha', '--bin_gate_prior_alpha', dest='cap_gate_prior_alpha', type=float, default=0.8,
                    help='strength of detached base-confidence prior for CAP gates')
parser.add_argument('--cap_rescue_context_alpha', '--bin_rescue_context_alpha', dest='cap_rescue_context_alpha', type=float, default=0.5,
                    help='context penalty in CAP rescue quality')
parser.add_argument('--cap_rescue_iou_alpha', '--bin_rescue_iou_alpha', dest='cap_rescue_iou_alpha', type=float, default=0.2,
                    help='CAP IoU contribution in CAP rescue quality')
parser.add_argument('--cap_rescue_ambiguity_alpha', '--bin_rescue_ambiguity_alpha', dest='cap_rescue_ambiguity_alpha', type=float, default=0.7,
                    help='ambiguity-aware modulation strength for CAP rescue gates')

parser.add_argument('--cpm_bank_size', '--pcm_bank_size', '--cspm_bank_size', dest='cpm_bank_size', type=int, default=64,
                    help='number of proposal supports stored per class in CPM')
parser.add_argument('--cpm_topk', '--pcm_topk', '--cspm_topk', dest='cpm_topk', type=int, default=4,
                    help='number of positive CPM memory slots retrieved per class')
parser.add_argument('--cpm_embed_dim', '--pcm_embed_dim', '--cspm_embed_dim', dest='cpm_embed_dim', type=int, default=256,
                    help='embedding dimension of CPM queries and supports')
parser.add_argument('--cpm_temperature', '--pcm_temperature', '--cspm_temperature', dest='cpm_temperature', type=float, default=0.07,
                    help='temperature for CPM support retrieval')
parser.add_argument('--cpm_residual_alpha', '--pcm_residual_alpha', '--cspm_residual_alpha', dest='cpm_residual_alpha', type=float, default=1.0,
                    help='residual strength of CPM edits on proposal classification logits')
parser.add_argument('--cpm_warmup_epoch', '--pcm_warmup_epoch', '--cspm_warmup_epoch', dest='cpm_warmup_epoch', type=int, default=20,
                    help='number of warmup epochs before CPM residual MIL is enabled')
parser.add_argument('--cpm_momentum', '--pcm_momentum', dest='cpm_momentum', type=float, default=0.995,
                    help='EMA momentum used by the CPM target encoder')

# training parameters
parser.add_argument('--batch_size', type=int, default=10, help='batch size')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
parser.add_argument('--max_epoch', type=int, default=200, help='maximum training epochs')
parser.add_argument('--rampup_length', type=int, default=30, help='consistency rampup length')
parser.add_argument('--interval', type=int, default=10, help='test interval in epochs')
parser.add_argument('--k', type=float, default=8, help='top-k divisor for video-level aggregation')
parser.add_argument('--gamma', type=float, default=0.8, help='threshold for selecting pseudo instances')

parser.add_argument('--weight_loss_prop_mil_orig', type=float, default=2, help='weight of original proposal MIL loss')
parser.add_argument('--weight_loss_prop_mil_supp', type=float, default=1, help='weight of attention-suppressed proposal MIL loss')
parser.add_argument('--weight_loss_prop_mil_refined', type=float, default=0.5, help='weight of CAP-refined proposal MIL loss')
parser.add_argument('--weight_loss_prop_irc', type=float, default=2, help='weight of instance rank consistency loss')
parser.add_argument('--weight_loss_prop_comp', type=float, default=20, help='weight of base proposal completeness loss')
parser.add_argument('--weight_loss_cap_comp', '--weight_loss_prop_cap_comp', '--weight_loss_prop_bin_comp',
                    dest='weight_loss_cap_comp', type=float, default=5.0, help='weight of CAP residual completeness loss')
parser.add_argument('--weight_loss_cap_rescue', '--weight_loss_prop_cap_rescue', '--weight_loss_prop_bin_rescue',
                    dest='weight_loss_cap_rescue', type=float, default=1.0, help='weight of CAP positive rescue loss')
parser.add_argument('--weight_loss_cap_action_dominance', '--weight_loss_prop_action_dominance',
                    dest='weight_loss_cap_action_dominance', type=float, default=1.0,
                    help='weight of CAP action-dominance loss')
parser.add_argument('--weight_loss_mil_cpm', '--weight_loss_prop_mil_cpm', '--weight_loss_prop_mil_pcm',
                    '--weight_loss_prop_mil_cspm', dest='weight_loss_mil_cpm', type=float, default=0.3,
                    help='weight of CPM-enhanced proposal MIL loss')
parser.add_argument('--weight_loss_cpm_cls', '--weight_loss_prop_cpm_cls', '--weight_loss_prop_pcm_cls',
                    '--weight_loss_prop_cspm_cls', dest='weight_loss_cpm_cls', type=float, default=0.3,
                    help='weight of CPM positive-support classification loss')
parser.add_argument('--weight_loss_cpm_rank', '--weight_loss_prop_cpm_rank', '--weight_loss_prop_pcm_rank',
                    '--weight_loss_prop_cspm_rank', dest='weight_loss_cpm_rank', type=float, default=0.3,
                    help='weight of CPM positive-support ranking loss')

parser.add_argument('--dominance_topk', type=int, default=5, help='top 1/k high-confidence proposals for CAP dominance')
parser.add_argument('--dominance_margin', type=float, default=0.1, help='margin for CAP action dominance')
parser.add_argument('--cap_rescue_topk', '--bin_rescue_topk', dest='cap_rescue_topk', type=int, default=6,
                    help='top-k under-scored positive proposals per class for CAP rescue')
parser.add_argument('--cap_rescue_margin', '--bin_rescue_margin', dest='cap_rescue_margin', type=float, default=0.05,
                    help='margin for CAP positive rescue ranking')
parser.add_argument('--cpm_seed_gamma', '--pcm_seed_gamma', '--cspm_seed_gamma', dest='cpm_seed_gamma', type=float, default=0.7,
                    help='confidence ratio for writing positive proposals into CPM memory')
parser.add_argument('--cpm_update_topk', '--pcm_update_topk', '--cspm_update_topk', dest='cpm_update_topk', type=int, default=2,
                    help='number of proposals per class and video used to update CPM memory')
parser.add_argument('--cpm_anchor_topk_div', '--pcm_anchor_topk_div', '--cspm_anchor_topk_div',
                    dest='cpm_anchor_topk_div', type=int, default=8,
                    help='top 1/k pseudo-positive proposals per class used in CPM supervision')
parser.add_argument('--cpm_rank_margin', '--pcm_rank_margin', '--cspm_rank_margin', dest='cpm_rank_margin', type=float, default=0.08,
                    help='margin for CPM positive-support ranking')

# testing parameters
parser.add_argument('--threshold_cls', type=float, default=0.2, help='video-level classification threshold')
parser.add_argument('--gamma_vid', type=float, default=0.0, help='contribution of video-level score to proposal score')
parser.add_argument('--dominance_alpha', type=float, default=1, help='residual strength of CAP dominance in proposal scoring')
parser.add_argument('--cpm_score_alpha', '--pcm_score_alpha', '--cspm_score_alpha', dest='cpm_score_alpha', type=float, default=1,
                    help='residual strength of CPM confidence bonus in proposal scoring')
