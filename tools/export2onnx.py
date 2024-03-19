# --------------------------------------------------------
# RepVGG: Making VGG-style ConvNets Great Again (https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf)
# Github source: https://github.com/DingXiaoH/RepVGG
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import argparse
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch import nn
from repvggplus import create_RepVGGplus_by_name

from loguru import logger


parser = argparse.ArgumentParser(description='RepVGG(plus) export to ONNX format')
parser.add_argument('-c', '--checkpoint', metavar='CKPT', default=None, type=str, help='path to the checkpoint weights file')
parser.add_argument('-a', '--arch', metavar='ARCH', default='RepVGG-A0')
parser.add_argument('--img-shape', nargs=2, type=int, default=[224, 224], help="image shape for export")
parser.add_argument("--output-name", default=None, type=str, help="output name of onnx model")
parser.add_argument("--input", default="images", type=str, help="input node name of onnx model")
parser.add_argument("--output", default="output", type=str, help="output node name of onnx model")
parser.add_argument("-o", "--opset", default=11, type=int, help="onnx opset version")
parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
parser.add_argument('--batch-size', default=1, type=int, help="batch size for exported onnx")


@logger.catch
def export():   
    args = parser.parse_args()
    logger.info("args value: {}".format(args))

    model = create_RepVGGplus_by_name(args.arch, deploy=False)
    output_name = args.output_name if args.output_name else args.arch + '.onnx'

    if args.checkpoint and os.path.isfile(args.checkpoint):
        logger.info("Loading checkpoint from: {}".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif 'model' in checkpoint:
            checkpoint = checkpoint['model']
        ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
        model.load_state_dict(ckpt)
    else:
        if args.checkpoint:
            logger.info("No checkpoint found at {}. --> Using random weights".format(args.checkpoint))
        else:
            logger.info("No checkpoint found --> Using random weights")
        model.apply(init_weights)
        output_name = output_name.replace(".onnx", "_random.onnx")

    model.eval()

    # RepVGG deployment
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()

    dummy_input = torch.randn(args.batch_size, 3, args.img_shape[0], args.img_shape[1])
    # Dry run
    dummy_output = model(dummy_input)

    torch.onnx._export(
        model,
        dummy_input,
        output_name,
        input_names=[args.input],
        output_names=[args.output],
        opset_version=args.opset,
    )
    logger.info("Generated onnx model named {}".format(output_name))

    if not args.no_onnxsim:
        import onnx
        from onnxsim import simplify

        # use onnx-simplifier to reduce reduent model.
        onnx_model = onnx.load(output_name)
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, output_name)
        logger.info("Generated simplified onnx model named {}".format(output_name))


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        # Initialize weights using Xavier (Glorot) uniform initialization
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


if __name__ == '__main__':
    export()
