import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def BGR_to_RGB(cvimg):
    pilimg = cvimg.copy()
    pilimg[:, :, 0] = cvimg[:, :, 2]
    pilimg[:, :, 2] = cvimg[:, :, 0]
    return pilimg



def test_mask(heat, save_path):
    # heat:[b, c, h, w]
    fig = plt.figure(figsize=(1, 1))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)

    for i in range(len(heat)):
        heat1 = heat[i]  # [c, h, w]
        heat1 = heat1.data.cpu().numpy()	     # 将tensor格式的feature map转为numpy格式
        # print(heat1.shape)
        # heat = np.squeeze(heat, 0)	         # ０维为batch维度，由于是单张图片，所以batch=1，将这一维度删除

        plt.subplot(1, 1, 1)
        plt.axis('off')

        channel_heat = heat1[0, :]     # [h, w]切片获取某几个通道的特征图
        heatmap = np.maximum(channel_heat, 0)        # heatmap与0比较

        # 用cv2加载原始图像
        img = cv2.imread('/media/newamax/94d146aa-e21d-4f2d-ae9d-1f5444870820/wyting/projects/SMENet_res_FAM2/VOCNWPU/VOC2012/JPEGImages/254.jpg')
        img = cv2.resize(img, (400, 400))
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 特征图的大小调整为与原始图像相同
        heatmap = np.uint8(255 * heatmap)  # 将特征图转换为uin
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将特征图转为伪彩色图
        heat_img = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)     # 将伪彩色图与原始图片融合
        #heat_img = heatmap * 0.5 + img 　　　　　　		  　　　 # 也可以用这种方式融合

        make_dirs(os.path.join(save_path, "per_channel_heatmap"))
        cv2.imwrite(save_path + "/per_channel_heatmap/channel_" + str(0) + ".jpg", heat_img)  # 将图像保存

        show_fea = BGR_to_RGB(heat_img)
        plt.imshow(show_fea)

        fig.savefig(save_path + "/channels_heatmap.jpg", dpi=100)
        fig.clf()
        plt.close()




# heat 为某层的特征图，自己手动获取
def test_visual(heat, save_path):
    # heat:[b, c, h, w]
    fig = plt.figure(figsize=(32, 32))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)

    for i in range(len(heat)):
        heat1 = heat[i]  # [c, h, w]
        heat1 = heat1.data.cpu().numpy()	     # 将tensor格式的feature map转为numpy格式
        # print(heat1.shape)
        # heat = np.squeeze(heat, 0)	         # ０维为batch维度，由于是单张图片，所以batch=1，将这一维度删除

        for j in range(1024):  # 64 channels
            plt.subplot(32, 32, j + 1)
            plt.axis('off')

            channel_heat = heat1[j, :]     # [h, w]切片获取某几个通道的特征图
            heatmap = np.maximum(channel_heat, 0)        # heatmap与0比较

            # 用cv2加载原始图像
            img = cv2.imread('/media/newamax/94d146aa-e21d-4f2d-ae9d-1f5444870820/wyting/projects/SMENet_res_FAM/VOCNWPU/VOC2012/JPEGImages/254.jpg')
            img = cv2.resize(img, (400, 400))
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 特征图的大小调整为与原始图像相同
            heatmap = np.uint8(255 * heatmap)  # 将特征图转换为uin
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将特征图转为伪彩色图
            heat_img = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)     # 将伪彩色图与原始图片融合
            #heat_img = heatmap * 0.5 + img 　　　　　　		  　　　 # 也可以用这种方式融合

            make_dirs(os.path.join(save_path, "per_channel_heatmap"))
            cv2.imwrite(save_path + "/per_channel_heatmap/channel_" + str(j) + ".jpg", heat_img)  # 将图像保存

            show_fea = BGR_to_RGB(heat_img)
            plt.imshow(show_fea)

        fig.savefig(save_path + "/channels_heatmap.jpg", dpi=100)
        fig.clf()
        plt.close()


def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def get_feature(outs, save_path):
    dst = './Visual_Features_detail'
    therd_size = 400

    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)

    for v in range(len(outs)):   # 一列表返回字典可遍历的键，值   [键，值]
        features = outs[0]   # [C, H, W]   => v[0,:,:,:]
        size = features.shape[1]
        for i in range(64):
        # for i in range(iter_range):
            plt.subplot(8, 8, i + 1)
            plt.axis('off')

            feature = features.cuda().data.cpu().numpy()
            feature_img = feature[i, :, :]  # 每一个通道的图   每一个通道显示为一幅独立的图
            feature_img = np.maximum(feature_img, 0)
            feature_img = np.asarray(feature_img * 255, dtype=np.uint8)
            # if size == 19:
            #     print(str(size) + '_' + str(i))
            #     print(feature[i,:,:])
            #     print(feature_img)

            dst_path = os.path.join(dst, str(size))
            make_dirs(dst_path)
            feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)

            # 对图片形状的控制
            if feature_img.shape[0] < therd_size:    # 对图片尺寸的控制
                tmp_file = os.path.join(dst_path, str(i) + '_' + str(therd_size) + '.png')
                tmp_img = feature_img.copy()
                tmp_img = cv2.resize(tmp_img, (therd_size, therd_size), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(tmp_file, tmp_img)

            dst_file = os.path.join(dst_path, str(i) + '.png')
            cv2.imwrite(dst_file, feature_img)

            show_fea = BGR_to_RGB(feature_img)
            plt.imshow(show_fea)

    fig.savefig(save_path, dpi=100)
    fig.clf()
    plt.close()
