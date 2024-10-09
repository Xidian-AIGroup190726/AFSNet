from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from AFSNet import build_SMENet
from sub_modules import *
import os
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import argparse
from tensorboardX import SummaryWriter


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='SMENet Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='resnet50.pth',         # backbone
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

# apply CUDA
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    if args.dataset == 'VOC':
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))

    # Building SMENet
    sme_net = build_SMENet('train', cfg['min_dim'], cfg['num_classes'])
    net = sme_net

    # Implementation parallel computing
    if args.cuda:
        print('cuda is avalible')
    """
    # Recover model from breakpoint
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        sme_net.load_weights(args.resume)

    # load backbone
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        sme_net.vgg.load_state_dict(vgg_weights)
    """
    if args.cuda:
        net = net.cuda()

    # If the model is not recovered from the breakpoint, the model is reinitialized
    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        # sme_net.extras.apply(weights_init)
        sme_net.loc.apply(weights_init)
        sme_net.conf.apply(weights_init)
        sme_net.change_channels.apply(weights_init)
        sme_net.Fusion_detailed_information1.apply(weights_init)
        sme_net.Fusion_detailed_information2.apply(weights_init)
        sme_net.Fusion_detailed_information3.apply(weights_init)
        sme_net.InLD1.apply(weights_init)
        sme_net.InLD2.apply(weights_init)
        sme_net.InLD3.apply(weights_init)

        for j in range(2):
            sme_net.Erase[j].apply(weights_init)

        for i in range(6):
            sme_net.FBS.apply(weights_init)

    if args.cuda:
        net.cuda()

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # loss
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.cuda)
    offset_criterion = Offset_loss()
    # Tensorboard visualization
    write = SummaryWriter('runs/SMENet')
    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    mask_loss = 0
    offset_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SMENet on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True, drop_last=True, generator=torch.Generator(device='cuda'))
    # create batch iterator
    batch_iterator = iter(data_loader)
    start_time = time.time()
    per_time = 0
    best_loss = 30
    iter_count = 0
    for iteration in range(args.start_iter, cfg['max_iter']):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            mask_loss = 0
            epoch += 1

        # Update learning rate
        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        try:
            images, targets = next(batch_iterator)  # targets:长度为batch的列表，每一个元素记录了一张图片中的目标[num_gt, 5] 5--->[xmin, ymin, xmax, ymax, cls_id]这里是相对坐标
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        # Using cuda
        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True)for ann in targets]

        # Extract each picture and its corresponding label from batch pictures and labels
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # out[0]:预测的偏移量[b, num_anchors, 4]  out[1]:预测的类别得分[b, num_anchors, 11]  out[2]:anchors坐标[num_anchors, 4]  out[3]:预测得到的mask
        # backprob
        optimizer.zero_grad()   # 梯度清零
        loss_l, loss_c, loss_m = criterion(out, targets)   # Calculate position loss and confidence loss

        # loss_f = offset_criterion(targets, sources)
        loss = loss_l + loss_c + loss_m     # 总损失
        # loss_cl = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()

        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        mask_loss += loss_m.item()
        # offset_loss += loss_f.item()

        write.add_scalar('Train_Loss', loss.item(), iteration)
        write.add_scalar('Location_Loss', loss_l.item(), iteration)
        write.add_scalar('Confidence_Loss', loss_c.item(), iteration)
        write.add_scalar('Mask_Loss', loss_m.item(), iteration)
        per_time += (t1 - t0)
        if iteration % 1 == 0:
            print('Spend time per 1000: %0.4f sec.' % (per_time))
            # print('iter:{0} ||Loss:{1:0.4f}||  ||Conf_Loss:{2:0.4f}||  ||Loc_Loss:{3:0.4f}||  ||Off_loss:{4:0.4f}|| \n'
            #       .format(iteration, loss.item(), loss_c.item(), loss_l.item(), loss_f.item()))
            print('iter:{0} ||Loss:{1:0.4f}||  ||Conf_Loss:{2:0.4f}||  ||Loc_Loss:{3:0.4f}||  ||Mask_Loss:{2:0.4f}||  \n'
                  .format(iteration, loss.item(), loss_c.item(), loss_l.item(), loss_m.item()))
            per_time = 0

        if loss.item() < best_loss:

            best_loss = loss.item()
            print('the Best initeration Now:'+ repr(iteration) +' '+ 'Loss: %.4f\n' % (loss.item()))
        if cfg['max_iter'] - iteration <= 5:
            torch.save(sme_net.state_dict(),
                       args.save_folder + '' + 'SME400' + str(iter_count) + '.pth')
            iter_count += 1

    write.close()
    end_time = time.time()
    print('Spend Total Time: %0.4f sec.' % (end_time - start_time))
    print('Best_loss:', best_loss)


# Adjust learning rate
def adjust_learning_rate(optimizer, gamma, step):
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        if m.bias is not None:
            xavier(m.weight.data)
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    train()
