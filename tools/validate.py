# reference:
# https://github.com/huggingface/pytorch-image-models/blob/v0.4.12/validate.py

""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import os
import csv
import glob
import json
import time
import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict
from contextlib import suppress

from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_legacy

has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True

qutntization_backend = 'fbgemm'
torch.backends.quantized.engine = qutntization_backend

_logger = logging.getLogger('validate')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--model', '-m', metavar='NAME', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--legacy-jit', dest='legacy_jit', action='store_true',
                    help='use legacy jit mode for pytorch 1.5/1.5.1/1.6 to get back fusion performance')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--real-labels', default='', type=str, metavar='FILENAME',
                    help='Real labels JSON file for imagenet evaluation')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')

parser.add_argument('--print-model', action='store_true',
                    help='Print model')
parser.add_argument('--replace-relu6', action='store_true',
                    help='Replace ReLU6 with ReLU activation')
parser.add_argument('--force-cpu', action='store_true',
                    help='Forcibly run inference on CPU')

parser.add_argument('--quant', dest='quantize', action='store_true',
                    help='Quantize model')
parser.add_argument('--layers-quantized', nargs='*', default=[])
parser.add_argument('--layers-not-quantized', nargs='*', default=[])
parser.add_argument('-cb', '--calib-batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size for caliblation (default: 64)')

parser.add_argument('-sa', '--sensitivity-analysis-targets', default=None,
                    help='Path to a JSON file describing sensitivity analysis target layer names')
parser.add_argument('-pq', '--partial-quantization', default=None,
                    help='Run partial quantization based on the specified sensitivity analysis result')
parser.add_argument('--worst-n', type=int, default=10,
                    help='Target of partial quantization')


def quantize(model, args, data_config, crop_pct):
    from torch.quantization import get_default_qconfig, quantize_fx

    dataset = create_dataset(
        root=args.data, name=args.dataset, split='calib',
        load_bytes=args.tf_preprocessing, class_map=args.class_map)

    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.calib_batch_size,  # too big batch size (e.g., 512) leads to non-deterministic result
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=crop_pct,
        pin_memory=args.pin_mem,
        tf_preprocessing=args.tf_preprocessing)
    
    qconfig = {
        '': get_default_qconfig(qutntization_backend),
    }
    assert len(args.layers_quantized) == 0 or len(args.layers_not_quantized) == 0
    if len(args.layers_quantized) > 0:
        qconfig = {
            '': None,
            'module_name': [(layer_name, get_default_qconfig(qutntization_backend)) for layer_name in args.layers_quantized],
        }
    if len(args.layers_not_quantized) > 0:
        qconfig = {
            '': get_default_qconfig(qutntization_backend),
            'module_name': [(layer_name, None) for layer_name in args.layers_not_quantized],
        }
    _logger.info(f'qconfig={qconfig}')
    example_inputs = torch.zeros((args.batch_size,) + tuple(data_config['input_size']))
    model = quantize_fx.prepare_fx(model.eval(), qconfig, example_inputs=example_inputs)

    for batch_idx, (input, target) in enumerate(loader):
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        target = target.cpu()
        input = input.cpu()

        # compute output
        with torch.no_grad():
            output = model(input)

    model = quantize_fx.convert_fx(model.eval())
    return model


def replace_relu6(model):
    reassign = {}
    for name, mod in model.named_children():
        replace_relu6(mod)
        # Checking for explicit type instead of instance
        # as we only want to replace modules of the exact type
        # not inherited classes
        if type(mod) == nn.ReLU or type(mod) == nn.ReLU6:
            reassign[name] = nn.ReLU(inplace=False)

    for key, value in reassign.items():
        model._modules[key] = value


def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher
    amp_autocast = suppress  # do nothing
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
            _logger.warning("Neither APEX or Native Torch AMP is available.")
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    if args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        _logger.info('Validating in mixed precision with native PyTorch AMP.')
    elif args.apex_amp:
        _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
    else:
        _logger.info('Validating in float32. AMP not enabled.')

    if args.legacy_jit:
        set_jit_legacy()

    # create model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=args.gp,
        scriptable=args.torchscript)
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(vars(args), model=model, use_test_size=True, verbose=True)
    test_time_pool = False
    if not args.no_test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config, use_test_size=True)

    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)

    if args.quantize and not args.force_cpu:
        args.force_cpu = True
        _logger.warning('PyTorch quantization does not support CUDA inference. Running inference in CPU mode.')

    if not args.force_cpu:
        model = model.cuda()
    if args.apex_amp:
        model = amp.initialize(model, opt_level='O1')

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    criterion = nn.CrossEntropyLoss()
    if not args.force_cpu:
        criterion = criterion.cuda()

    dataset = create_dataset(
        root=args.data, name=args.dataset, split=args.split,
        load_bytes=args.tf_preprocessing, class_map=args.class_map)

    if args.valid_labels:
        with open(args.valid_labels, 'r') as f:
            valid_labels = {int(line.rstrip()) for line in f}
            valid_labels = [i in valid_labels for i in range(args.num_classes)]
    else:
        valid_labels = None

    if args.real_labels:
        real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
    else:
        real_labels = None

    crop_pct = 1.0 if test_time_pool else data_config['crop_pct']
    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=crop_pct,
        pin_memory=args.pin_mem,
        tf_preprocessing=args.tf_preprocessing)

    if args.replace_relu6:
        replace_relu6(model)

    if args.quantize:
        _logger.info('Quantizing model...')
        model = quantize(model, args, data_config, crop_pct)
        _logger.info('Done quantization.')

    if args.print_model:
        print(model)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        input = torch.randn((args.batch_size,) + tuple(data_config['input_size'])).cuda()
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        if args.force_cpu:
            input = input.cpu()
        model(input)

        end = time.time()
        for batch_idx, (input, target) in enumerate(loader):
            if args.no_prefetcher and not args.force_cpu:
                target = target.cuda()
                input = input.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            if args.force_cpu:
                target = target.cpu()
                input = input.cpu()

            # compute output
            with amp_autocast():
                output = model(input)

            if valid_labels is not None:
                output = output[:, valid_labels]
            loss = criterion(output, target)

            if real_labels is not None:
                real_labels.add_result(output)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                    'Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})'.format(
                        batch_idx, len(loader), batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                        loss=losses, top1=top1, top5=top5))

    if real_labels is not None:
        # real labels mode replaces topk values at the end
        top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    else:
        top1a, top5a = top1.avg, top5.avg
    results = OrderedDict(
        top1=round(top1a, 4), top1_err=round(100 - top1a, 4),
        top5=round(top5a, 4), top5_err=round(100 - top5a, 4),
        param_count=round(param_count / 1e6, 2),
        img_size=data_config['input_size'][-1],
        cropt_pct=crop_pct,
        interpolation=data_config['interpolation'])

    _logger.info(' * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})'.format(
       results['top1'], results['top1_err'], results['top5'], results['top5_err']))

    return results


def main_sensitivity_analysis(args):
    assert len(args.layers_quantized) == 0
    assert len(args.layers_not_quantized) == 0

    with open(args.sensitivity_analysis_targets) as f:
        target_layers_list = json.load(f)['layers_quantized']

    _logger.info(f'Target layers of sensitivity analysis: {target_layers_list}')

    args.quantize = True
    results = []
    for idx, target_layers in enumerate(target_layers_list):
        _logger.info(f'Running sensitivity analysis of layer: {idx + 1}/{len(target_layers_list)} ..')
        args.layers_quantized = target_layers
        r = validate(args)
        r['layers_quantized'] = target_layers
        results.append(r)
        write_results(f'sensitivity_analysis_{args.model}.csv', results)


def main_partial_quantization(args):
    assert len(args.layers_quantized) == 0
    assert len(args.layers_not_quantized) == 0

    sa_df = pd.read_csv(args.partial_quantization)
    sa_df = sa_df.sort_values(by='top1').reset_index()

    target_layers_list = list(sa_df.layers_quantized)
    target_layers_list = [eval(x) for x in target_layers_list]
    target_layers_list = target_layers_list[:args.worst_n]
    target_layers_list.insert(0, [])

    args.quantize = True
    layers_not_quantized = []
    results = []
    for idx, target_layers in enumerate(target_layers_list):
        _logger.info(f'Running partial quantization of layer: {idx + 1}/{len(target_layers_list)} ..')
        layers_not_quantized.extend(target_layers)
        args.layers_not_quantized = layers_not_quantized
        r = validate(args)
        r['layers_not_quantized'] = target_layers
        results.append(r)
        write_results(f'partial_quantization_{args.model}.csv', results)


def main_validate(args):
    model_cfgs = []
    model_names = []
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + '/*.pth.tar')
        checkpoints += glob.glob(args.checkpoint + '/*.pth')
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == 'all':
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(pretrained=True, exclude_filters=['*_in21k', '*_in22k'])
            model_cfgs = [(n, '') for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(args.model)
            model_cfgs = [(n, '') for n in model_names]

    if len(model_cfgs):
        results_file = args.results_file or './results-all.csv'
        _logger.info('Running bulk validation on these pretrained models: {}'.format(', '.join(model_names)))
        results = []
        try:
            start_batch_size = args.batch_size
            for m, c in model_cfgs:
                batch_size = start_batch_size
                args.model = m
                args.checkpoint = c
                result = OrderedDict(model=args.model)
                r = {}
                while not r and batch_size >= args.num_gpu:
                    torch.cuda.empty_cache()
                    try:
                        args.batch_size = batch_size
                        print('Validating with batch size: %d' % args.batch_size)
                        r = validate(args)
                    except RuntimeError as e:
                        if batch_size <= args.num_gpu:
                            print("Validation failed with no ability to reduce batch size. Exiting.")
                            raise e
                        batch_size = max(batch_size // 2, args.num_gpu)
                        print("Validation failed, reducing batch size by 50%")
                result.update(r)
                if args.checkpoint:
                    result['checkpoint'] = args.checkpoint
                results.append(result)
        except KeyboardInterrupt as e:
            pass
        results = sorted(results, key=lambda x: x['top1'], reverse=True)
        if len(results):
            write_results(results_file, results)
    else:
        validate(args)


def main():
    setup_default_logging()
    args = parser.parse_args()
    if args.sensitivity_analysis_targets is not None:
        main_sensitivity_analysis(args)
    elif args.partial_quantization is not None:
        main_partial_quantization(args)
    else:
        main_validate(args)


def write_results(results_file, results):
    with open(results_file, mode='w') as cf:
        dw = csv.DictWriter(cf, fieldnames=results[0].keys())
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()


if __name__ == '__main__':
    main()
