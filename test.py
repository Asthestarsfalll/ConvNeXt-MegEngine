import argparse
import time

import megengine as mge
import megengine.data as data
import megengine.data.transform as T
import megengine.distributed as dist
import megengine.functional as F

from models.convnext import *
from utils import AverageMeter, build_dataset

logging = mge.logger.get_logger()

model_mapper = {
    "tiny": convnext_tiny,
    "samll": convnext_small,
    "base": convnext_base,
    "large": convnext_large,
    "xlarge": convnext_xlarge,
}

def make_parser():
    parser = argparse.ArgumentParser(description="MegEngine ImageNet Training")
    parser.add_argument("-d", "--data", metavar="DIR",
                        help="path to imagenet dataset")
    parser.add_argument(
        "-a",
        "--arch",
        default="tiny",
        help="model architecture (default: tiny)",
    )
    parser.add_argument(
        "-n",
        "--ngpus",
        default=None,
        type=int,
        help="number of GPUs per node (default: None, use all available GPUs)",
    )
    parser.add_argument(
        "-c", "--ckpt",
        default=None,
        help="path to model checkpoint"
    )
    parser.add_argument(
        "-j", "--workers",
        default=2,
        type=int
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=20,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )

    parser.add_argument("--val-batch-size", default=128, type=int)
    parser.add_argument("--dist-addr", default="localhost")
    parser.add_argument("--dist-port", default=23456, type=int)
    parser.add_argument("--world-size", default=-1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    if args.ngpus is None:
        args.ngpus = mge.get_device_count("gpu")

    if args.world_size * args.ngpus > 1:
        dist_worker = dist.launcher(
            master_ip=args.dist_addr,
            port=args.dist_port,
            world_size=args.world_size * args.ngpus,
            rank_start=args.rank * args.ngpus,
            n_gpus=args.ngpus
        )(worker)
        dist_worker(args)
    else:
        worker(args)


def worker(args):
    # build dataset
    print('build dataset')
    _, valid_dataloader = build_dataset(args, is_train=False)

    # build model
    print('build model')
    model = model_mapper[args.arch](True)
    # use pretrained model default
    if args.ckpt is not None:
        logging.info("load from checkpoint %s", args.ckpt)
        print("load from checkpoint %s", args.ckpt)
        checkpoint = mge.load(args.ckpt)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(checkpoint)

    def valid_step(image, label):
        logits = model(image)
        loss = F.nn.cross_entropy(logits, label)
        acc1, acc5 = F.topk_accuracy(logits, label, topk=(1, 5))
#         print(F.argmax(logits,axis=1))
#         print('111:', logits[:,111])
#         print(label)
        # calculate mean values
        if dist.get_world_size() > 1:
            loss = F.distributed.all_reduce_sum(loss) / dist.get_world_size()
            acc1 = F.distributed.all_reduce_sum(acc1) / dist.get_world_size()
            acc5 = F.distributed.all_reduce_sum(acc5) / dist.get_world_size()
        return loss, acc1, acc5

    model.eval()
    _, valid_acc1, valid_acc5 = valid(valid_step, valid_dataloader, args)
    logging.info(
        "Test Acc@1 %.3f, Acc@5 %.3f",
        valid_acc1,
        valid_acc5,
    )


def valid(func, data_queue, args):
    objs = AverageMeter("Loss")
    top1 = AverageMeter("Acc@1")
    top5 = AverageMeter("Acc@5")
    clck = AverageMeter("Time")

    t = time.time()
    print('start val')
    print(len(data_queue))
    
    for step, (image, label) in enumerate(data_queue):
        image = mge.tensor(image, dtype="float32")
        label = mge.tensor(label, dtype="int32")
        n = image.shape[0]

        loss, acc1, acc5 = func(image, label)

        objs.update(loss.item(), n)
        top1.update(100 * acc1.item(), n)
        top5.update(100 * acc5.item(), n)
        clck.update(time.time() - t, n)
        t = time.time()
        if step % args.print_freq == 0 and dist.get_rank() == 0:
            logging.info("Test step %d, %s %s %s %s",
                         step, objs, top1, top5, clck)

    return objs.avg, top1.avg, top5.avg


if __name__ == "__main__":
    main()
