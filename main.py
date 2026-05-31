import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import random
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import options
import utils
from dataset import VideoDataset
from eval_detection import getClassificationMAP as cmAP, ANETdetection
from model import P_MIL


def copy_source_files(backup_dir):
    """
    Back up Python source files without relying on shell-specific commands.
    """
    project_dir = Path(__file__).resolve().parent
    for src_file in project_dir.glob("*.py"):
        shutil.copy2(src_file, Path(backup_dir) / src_file.name)


def get_train_num_workers(args):
    """
    Use an explicit worker count so repeated runs keep the same data loading behavior.
    """
    return args.num_workers


def seed_worker(worker_id):
    """
    Seed DataLoader workers deterministically when num_workers > 0.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed):
    try:
        default_device = torch.empty((), dtype=torch.int64).device
        generator = torch.Generator(device=default_device)
    except TypeError:
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


LEGACY_STATE_KEY_REPLACEMENTS = (
    ('bin_embed', 'cap_embed'),
    ('bin_mixer', 'cap_mixer'),
    ('bin_classifier', 'cap_classifier'),
    ('bin_occupancy', 'cap_occupancy'),
    ('bin_class_occupancy', 'cap_class_occupancy'),
    ('bin_iou_occupancy', 'cap_iou_occupancy'),
    ('bin_cls_gate', 'cap_cls_gate'),
    ('bin_iou_gate', 'cap_iou_gate'),
    ('bin_delta_cls', 'cap_delta_cls'),
    ('bin_delta_iou', 'cap_delta_iou'),
    ('pcm.', 'cpm.'),
)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def remap_legacy_state_dict(state_dict):
    remapped = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith('module.') else key
        for old, new in LEGACY_STATE_KEY_REPLACEMENTS:
            new_key = new_key.replace(old, new)
        remapped[new_key] = value
    return remapped


def load_model_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = remap_legacy_state_dict(extract_state_dict(checkpoint))
    load_msg = model.load_state_dict(state_dict, strict=False)
    print('Loaded checkpoint: {}, missing keys: {}, unexpected keys: {}'.format(
        checkpoint_path, len(load_msg.missing_keys), len(load_msg.unexpected_keys)))
    if load_msg.missing_keys:
        print('  missing examples: {}'.format(load_msg.missing_keys[:5]))
    if load_msg.unexpected_keys:
        print('  unexpected examples: {}'.format(load_msg.unexpected_keys[:5]))
    return load_msg


def choose_test_checkpoint(args):
    if args.pretrained_ckpt:
        return args.pretrained_ckpt

    candidates = (
        Path(args.exp_dir) / 'ckpt' / 'best_model.pkl',
        Path('ckpt') / 'best_model.pkl',
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError('No test checkpoint found. Set --pretrained_ckpt explicitly.')


def train_one_epoch(epoch, args, dataset, model, optimizer, logger, device):
    """
    Train the model for one epoch on the training set.
    """
    model.train()
    loss_dict_sum = {}

    # Use a local RNG and do not mutate dataset.trainidx in-place.
    indices_train = list(dataset.trainidx)
    random.Random(args.seed + epoch).shuffle(indices_train)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=args.batch_size,
                              num_workers=get_train_num_workers(args),
                              drop_last=True,
                              pin_memory=device.type == 'cuda',
                              sampler=indices_train,
                              worker_init_fn=seed_worker,
                              generator=make_generator(args.seed + epoch),
                              collate_fn=utils.collate_fn)

    for step, sample in enumerate(train_loader):
        features = sample['features']
        proposals = sample['proposals']
        labels = sample['labels']
        video_indices = torch.tensor(sample['video_index'], dtype=torch.long, device=device)

        features = [torch.from_numpy(feat).float().to(device) for feat in features]
        proposals = [torch.from_numpy(prop).float().to(device) for prop in proposals]
        labels = [torch.from_numpy(label).float().to(device) for label in labels]
        labels = torch.stack(labels, dim=0)

        outputs = model(features, proposals, video_indices=video_indices)
        loss_dict = model.criterion(outputs, labels, proposals, epoch=epoch, args=args)

        for key in loss_dict.keys():
            if key not in loss_dict_sum.keys():
                loss_dict_sum[key] = 0
            loss_dict_sum[key] += loss_dict[key].cpu().item()

        loss_total = loss_dict['loss_total']
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        if hasattr(model, 'update_cpm_teacher'):
            model.update_cpm_teacher(args.cpm_momentum)
        if hasattr(model, 'update_cpm_memory'):
            model.update_cpm_memory(outputs, labels, epoch, args)

    print('Epoch: {}, Loss: {:.5f}'.format(epoch, loss_dict_sum['loss_total'] / len(train_loader)))
    for key in loss_dict_sum.keys():
        logger.add_scalar('loss/'+key, loss_dict_sum[key] / len(train_loader), epoch)


def train(args, dataset, model, device):
    """
    Train and test the model on the given dataset.
    """
    seed = args.seed
    print('=============seed: {}, pid: {}============='.format(seed, os.getpid()))
    utils.setup_seed(seed)

    parser = options.parser
    exp_dir = args.exp_dir
    os.makedirs(exp_dir, exist_ok=True)
    utils.print_options(args, parser, exp_dir)

    log_dir = os.path.join(args.exp_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger = SummaryWriter(log_dir)

    backup_dir = os.path.join(args.exp_dir, 'code_backup')
    os.makedirs(backup_dir, exist_ok=True)
    copy_source_files(backup_dir)

    checkpoint_dir = os.path.join(args.exp_dir, 'ckpt')
    os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    max_map = np.zeros(7) if 'Thumos' in args.dataset_name else np.zeros(10)
    for epoch in range(1, args.max_epoch+1):
        if hasattr(model, 'reset_cpm_memory'):
            model.reset_cpm_memory()
        train_one_epoch(epoch, args, dataset, model, optimizer, logger, device)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'last_model.pkl'))

        # evaluate model and save best model
        if epoch % args.interval == 0:
            iou, dmap = test(epoch, args, dataset, model, logger, device)
            cond = np.mean(dmap[:7]) > np.mean(max_map[:7]) if 'Thumos' in args.dataset_name else np.mean(dmap) > np.mean(max_map)
            if cond:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pkl'))
                max_map = dmap

    load_model_checkpoint(model, os.path.join(checkpoint_dir, 'best_model.pkl'), device)
    iou, dmap = test('best', args, dataset, model, logger, device)


@torch.no_grad()
def test(epoch, args, dataset, model, logger, device):
    """
    Test the model on the test set.

    Returns:
        iou (numpy.ndarray): Array of IoU thresholds.
        dmap (numpy.ndarray): Detection mean average precision (mAP) at different IoU thresholds.
    """
    model.eval()

    pred_vid_stack = []
    labels_stack = []
    predictions = []
    num_correct = 0

    indices_test = list(dataset.testidx)
    test_loader = DataLoader(dataset=dataset,
                             batch_size=1,
                             drop_last=False,
                             pin_memory=device.type == 'cuda',
                             num_workers=0,
                             sampler=indices_test,
                             collate_fn=utils.collate_fn)

    for step, sample in enumerate(test_loader):
        videoname = sample['videonames'][0]
        features = sample['features']
        proposals = sample['proposals']
        labels = sample['labels'][0]
        video_indices = torch.tensor([sample['video_index'][0]], dtype=torch.long, device=device)

        features = [torch.from_numpy(feat).float().to(device) for feat in features]
        proposals = [torch.from_numpy(prop).float().to(device) for prop in proposals]

        outputs = model(features, proposals, is_training=False, video_indices=video_indices)
        prediction, pred, pred_vid_score = utils.get_prediction(videoname, outputs, dataset, args)
        predictions.append(prediction)

        # calculate the number of correct video-level category predictions
        pred_np = np.zeros_like(labels)
        pred_np[pred] = 1
        correct_pred = np.sum(pred_np == labels)
        num_correct += (correct_pred == len(labels))

        pred_vid_stack.append(pred_vid_score)
        labels_stack.append(labels)

    pred_vid_stack = np.array(pred_vid_stack)
    labels_stack = np.array(labels_stack)

    # calculate the mean Average Precision (mAP) for each IoU threshold
    iou = np.linspace(0.1, 0.7, 7)
    dmap_detect = ANETdetection(dataset.path_to_annotations, tiou_thresholds=iou, subset="test", verbose=True)
    dmap_detect.prediction = pd.concat(predictions).reset_index(drop=True)
    dmap, dmap_class = dmap_detect.evaluate()   # dmap_class: [len(iou), n_class]

    test_acc = num_correct / len(test_loader) * 100
    cmap = cmAP(pred_vid_stack, labels_stack)
    print('Classification mAP {:5.2f}'.format(cmap))
    print('Classification acc {:5.2f}'.format(test_acc))

    if args.run_type == 'train':
        if not epoch == 'best':
            logger.add_scalar('test_mAP/Classification mAP', cmap, epoch)
            logger.add_scalar('test_mAP/Classification acc', test_acc, epoch)
            for item in list(zip(dmap, iou)):
                logger.add_scalar('test_mAP/Detection mAP @ IoU = ' + str(item[1]), item[0], epoch)
            logger.add_scalar('test_mAP/Detection mAP @ IoU = 0.1 : 0.5', np.mean(dmap[:5]), epoch)
            logger.add_scalar('test_mAP/Detection mAP @ IoU = 0.1 : 0.7', np.mean(dmap[:7]), epoch)
            logger.add_scalar('test_mAP/Detection mAP @ IoU = 0.3 : 0.7', np.mean(dmap[2:]), epoch)
        utils.write_to_file(args.exp_dir, dmap, cmap, test_acc, epoch)
    return iou, dmap


if __name__ == '__main__':
    args = options.parser.parse_args()
    utils.setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.run_type == 'test':
        args.pretrained_ckpt = choose_test_checkpoint(args)

    model = P_MIL(args).to(device)
    if args.pretrained_ckpt is not None:
        load_model_checkpoint(model, args.pretrained_ckpt, device)
    dataset = VideoDataset(args)

    if args.run_type == 'train':
        dataset.get_proposals(only_test=False)
        train(args, dataset, model, device)
    else:
        dataset.get_proposals(only_test=True)
        test('best', args, dataset, model, None, device)
