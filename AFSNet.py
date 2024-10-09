from torch.autograd import Variable
from layers import *
from data import voc
from sub_modules import *
from res50_backbone import resnet50
from visual_featuremap import *
from draw_feachermap import draw_features


class SMENet(nn.Module):
    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SMENet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = voc
        self.priorbox = PriorBox(self.cfg)    # return prior_box[cx,cy,w,h]
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size
        self.blocks_fusion = [4, 2, 1]
        self.vgg = base
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])    # Location_para
        self.conf = nn.ModuleList(head[1])   # Confidence_Para

        self.resnet_fusion = ResNet_fusion(self.blocks_fusion)
        self.change_channels = Change_channels()
        self.Erase = nn.ModuleList(Erase())
        self.FBS = nn.ModuleList(OSE_())
        self._init_weights()

        self.Fusion_detailed_information1 = Fusion_detailed_information1()
        self.Fusion_detailed_information2 = Fusion_detailed_information2(in_places=1024, places=512)
        self.Fusion_detailed_information3 = Fusion_detailed_information3(in_places=1024, places=512)

        self.InLD1 = InLD(256, 1, self.num_classes)
        self.InLD2 = InLD(256, 1, self.num_classes)
        self.InLD3 = InLD(256, 1, self.num_classes)

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def _init_weights(self):
        layers = [*self.extras]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def forward(self, x):
        sources = list()    # save detected feature maps
        loc = list()
        conf = list()

        x = self.vgg(x)
        sources.append(x)

        for layer in self.extras:
            x = layer(x)
            sources.append(x)

        save_path = "./feature_map_save"

        # draw_features(16, 16, sources[0].cpu().numpy(), "{}/original_feature_1".format(save_path))
        # draw_features(16, 16, sources[1].cpu().numpy(), "{}/original_feature_2".format(save_path))
        # draw_features(16, 16, sources[2].cpu().numpy(), "{}/original_feature_3".format(save_path))
        # draw_features(16, 16, sources[3].cpu().numpy(), "{}/original_feature_4".format(save_path))
        # draw_features(16, 16, sources[4].cpu().numpy(), "{}/original_feature_5".format(save_path))
        # draw_features(16, 16, sources[5].cpu().numpy(), "{}/original_feature_6".format(save_path))

        # visual(sources[0], './Visual_Features/ori_feature1')
        # get_feature(sources[0], "./Visual_Features/ori_fea_1.jpg")
        # visual(sources[1], './Visual_Features/ori_feature2')
        # get_feature(sources[1], "./Visual_Features/ori_fea_2.jpg")
        # visual(sources[2], './Visual_Features/ori_feature3')
        # get_feature(sources[2], "./Visual_Features/ori_fea_3.jpg")
        # visual(sources[3], './Visual_Features/ori_feature4')
        # get_feature(sources[3], "./Visual_Features/ori_fea_4.jpg")

        # Eliminate irrelevant information
        erase_sources0 = self.Erase[0](sources[0], sources[2])  # p1'
        erase_sources1 = self.Erase[1](sources[1], sources[3])  # p2'
        # Transmit detailed information
        sources[2] = self.Fusion_detailed_information3(sources[0], sources[1], sources[2])   #1024
        sources[1] = self.Fusion_detailed_information2(sources[0], erase_sources1)   # 1024
        sources[0] = self.Fusion_detailed_information1(erase_sources0)   # 1024

        sources[0], sources[1], sources[2] = self.change_channels(sources[0], sources[1], sources[2])

        mask1, sources[0] = self.InLD1(sources[0])
        mask2, sources[1] = self.InLD2(sources[1])
        mask3, sources[2] = self.InLD3(sources[2])

        mask = [mask1, mask2, mask3]

        # draw_features(16, 16, sources[0].cpu().numpy(), "{}/final_feature_1".format(save_path))
        # draw_features(16, 16, sources[1].cpu().numpy(), "{}/final_feature_2".format(save_path))
        # draw_features(16, 16, sources[2].cpu().numpy(), "{}/final_feature_3".format(save_path))
        # draw_features(16, 16, sources[3].cpu().numpy(), "{}/final_feature_4".format(save_path))
        # draw_features(16, 16, sources[4].cpu().numpy(), "{}/final_feature_5".format(save_path))
        # draw_features(16, 16, sources[5].cpu().numpy(), "{}/final_feature_6".format(save_path))
        # visual(sources[0], './Visual_Features/final_feature1')
        # get_feature(sources[0], "./Visual_Features/final_fea_1.jpg")
        # visual(sources[1], './Visual_Features/final_feature2')
        # get_feature(sources[1], "./Visual_Features/final_fea_2.jpg")
        # visual(sources[2], './Visual_Features/final_feature3')
        # get_feature(sources[2], "./Visual_Features/final_fea_3.jpg")
        # visual(sources[3], './Visual_Features/final_feature4')
        # get_feature(sources[3], "./Visual_Features/final_fea_4.jpg")

        # objects saliency enhancement
        # for i in range(len(sources)):
        #     sources[i] = self.FBS[i](sources[i])

        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = self.detect.forward(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors,
                mask
            )
        return output

    # load weights
    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage), strict=False)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


class Backbone(nn.Module):
    def __init__(self, pretrain_path=None):
        super(Backbone, self).__init__()
        net = resnet50()

        # torch.load()  加载的是训练好的模型
        # load_state_dict()是net的一个方法,是将torch.load加载出来的数据加载到net中
        if pretrain_path is not None:
            net.load_state_dict(torch.load(pretrain_path))

        self.feature_extractor = nn.Sequential(*list(net.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]

        # 修改conv4_block1的步距，从2->1  (水平方向步距, 竖直方向步距)
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


def add_extras(input_size):
    additional_blocks = []
    # input_size = [1024, 1024, 512, 256, 256, 256] for resnet50
    middle_channels = [256, 256, 128, 128, 128]
    for i, (input_ch, output_ch, middle_ch) in enumerate(zip(input_size[:-1], input_size[1:], middle_channels)):
        padding, stride = (1, 2) if i < 3 else (0, 1)
        layer = nn.Sequential(
            nn.Conv2d(input_ch, middle_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_ch, output_ch, kernel_size=3, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(inplace=True),
        )
        additional_blocks.append(layer)
    additional_blocks = nn.ModuleList(additional_blocks)
    return additional_blocks


def multibox(vgg, extra_layers, cfg, num_classes):
    conf_layer = []
    loc_layer = []
    for i in range(len(cfg)):
        conf_layer.append(
            nn.Conv2d(in_channels=256, out_channels=cfg[i] * num_classes, kernel_size=3, stride=1, padding=1))
        loc_layer.append(nn.Conv2d(in_channels=256, out_channels=cfg[i] * 4, kernel_size=3, padding=1, stride=1))
    return vgg, extra_layers, (loc_layer, conf_layer)


mbox = {
    '400': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


out_channels = [1024, 1024, 512, 256, 256, 256]


# External call interface function
def build_SMENet(phase, size=400, num_classes=11):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 400:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only  (size=300) is supported!")
        return
    resnet_weights_path = "./weights/resnet50.pth"
    base_, extras_, head_ = multibox(Backbone(resnet_weights_path), add_extras(out_channels), mbox[str(size)],
                                     num_classes)

    return SMENet(phase, size, base_, extras_, head_, num_classes)

