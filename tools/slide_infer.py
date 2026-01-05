import argparse
import os
import sys

sys.path.append(os.getcwd())

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor


class LoadImage:
    """A simple pipeline to load a single image."""

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


class LoadImagePair:
    """A simple pipeline to load a pair of images and concatenate channels."""

    def __call__(self, results):
        img1_path, img2_path = results['img']
        results['filename'] = img1_path
        results['ori_filename'] = img1_path
        img1 = mmcv.imread(img1_path)
        img2 = mmcv.imread(img2_path)
        if img1.shape[:2] != img2.shape[:2]:
            raise ValueError(
                f'Input images must share the same spatial size, '
                f'got {img1.shape[:2]} vs {img2.shape[:2]}.'
            )
        img = np.concatenate((img1, img2), axis=-1)
        results['img'] = img
        results['img_shape'] = img1.shape
        results['ori_shape'] = img1.shape
        return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sliding-window inference for large images',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            'Examples:\n'
            '  Single image:\n'
            '    python tools/slide_infer.py configs/foo.py work_dir/latest.pth \\\n'
            '      data/val/img.png --crop-size 1024 1024 --stride 768 768 \\\n'
            '      --out outputs/seg.png\n'
            '  Change detection (paired inputs):\n'
            '    python tools/slide_infer.py configs/cd.py work_dir/latest.pth \\\n'
            '      data/img_t1.png --img2 data/img_t2.png \\\n'
            '      --crop-size 1024 1024 --stride 768 768 --out outputs/cd.npy\n'
        ),
    )
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('img', help='Input image path')
    parser.add_argument(
        '--img2',
        default=None,
        help='Second input image path for change detection models')
    parser.add_argument(
        '--out',
        default=None,
        help='Output path (.png/.jpg for visualization, .npy/.pkl for labels)')
    parser.add_argument(
        '--device', default='cuda:0', help='CUDA device or cpu')
    parser.add_argument(
        '--crop-size',
        nargs=2,
        type=int,
        metavar=('H', 'W'),
        help='Sliding window crop size (height width)')
    parser.add_argument(
        '--stride',
        nargs=2,
        type=int,
        metavar=('H', 'W'),
        help='Sliding window stride (height width)')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity for visualization output')
    return parser.parse_args()


def build_model(cfg, checkpoint, device):
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = cfg
    model.to(device)
    model.eval()
    return model


def ensure_slide_cfg(cfg, crop_size, stride):
    test_cfg = cfg.get('test_cfg', {})
    if crop_size is not None:
        test_cfg['crop_size'] = tuple(crop_size)
    if stride is not None:
        test_cfg['stride'] = tuple(stride)
    if test_cfg.get('mode', 'slide') != 'slide':
        test_cfg['mode'] = 'slide'
    if 'crop_size' not in test_cfg or 'stride' not in test_cfg:
        raise ValueError(
            'Sliding-window inference requires crop_size and stride. '
            'Provide them in config test_cfg or via --crop-size/--stride.'
        )
    cfg.test_cfg = test_cfg
    return cfg


def build_data(cfg, img_path, img2_path=None):
    if img2_path is None:
        test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
        data = dict(img=img_path)
    else:
        test_pipeline = [LoadImagePair()] + cfg.data.test.pipeline[1:]
        data = dict(img=(img_path, img2_path))
    test_pipeline = Compose(test_pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    return data


def save_result(model, img_path, result, out_path, opacity):
    if out_path is None:
        return
    if out_path.endswith(('.png', '.jpg', '.jpeg')):
        model.show_result(
            img_path,
            result,
            index=0,
            out_file=out_path,
            opacity=opacity,
        )
        return
    pred = result[0]
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if out_path.endswith('.npy'):
        np.save(out_path, pred)
    else:
        mmcv.dump(pred, out_path)


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    cfg = ensure_slide_cfg(cfg, args.crop_size, args.stride)
    model = build_model(cfg, args.checkpoint, args.device)

    data = build_data(cfg, args.img, args.img2)
    if next(model.parameters()).is_cuda:
        data = scatter(data, [next(model.parameters()).device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    save_result(model, args.img, result, args.out, args.opacity)


if __name__ == '__main__':
    main()
