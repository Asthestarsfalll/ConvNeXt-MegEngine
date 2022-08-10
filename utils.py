import cv2
import megengine.data as data
import megengine.data.transform as T


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_train_trans(args):
    normalize = T.Normalize(
        mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]  # BGR
    )
    if (not hasattr(args, 'resolution')) or args.resolution == 224:
        trans = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(0.5),
            normalize,
            T.ToMode("CHW")
        ])
    else:
        raise ValueError('Not yet implemented.')
    return trans


def get_val_trans(args):
    normalize = T.Normalize(
        mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]  # BGR
    )
    if (not hasattr(args, 'resolution')) or args.resolution == 224:
        trans = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            normalize,
            T.ToMode("CHW")
        ])
    else:
        trans = T.Compose([
            T.Resize(
                args.resolution, interpolation=cv2.INTER_LINEAR),
            T.CenterCrop(args.resolution),
            normalize,
            T.ToMode("CHW")
        ])
    return trans


def build_dataset(args, is_train=True):
    if is_train:
        train_trans = get_train_trans(args)
        train_dataset = data.dataset.ImageNet(args.data, train=True)
        train_sampler = data.Infinite(
            data.RandomSampler(
                train_dataset, batch_size=args.batch_size, drop_last=True)
        )
        train_dataloader = data.DataLoader(
            train_dataset,
            sampler=train_sampler,
            transform=train_trans,
            num_workers=args.workers
        )
    else:
        train_dataloader = None

    val_trans = get_val_trans(args)
    valid_dataset = data.dataset.ImageNet(args.data, train=False)
    valid_sampler = data.SequentialSampler(
        valid_dataset, batch_size=args.val_batch_size, drop_last=False
    )
    valid_dataloader = data.DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        transform=val_trans,
        num_workers=args.workers
    )
    return train_dataloader, valid_dataloader

