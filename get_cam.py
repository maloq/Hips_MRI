import os
import argparse
import numpy as np

import torch

import torch.nn.functional as F

from dataloader import MRDataset
import model
import cv2


def return_CAM(feature_conv, weight, class_idx):
    # generate the class -activation maps upsample to 256x256
    size_upsample = (256, 256)
    feature_conv = np.expand_dims(feature_conv, axis=0)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        beforeDot = feature_conv.reshape((nc, h * w))
        cam = np.matmul(weight[idx], beforeDot)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def evaluate_model(model, val_loader, weights):
    if torch.cuda.is_available():
        model.cuda()
    predicted_labels = []

    for i, (image, label, weight) in enumerate(val_loader):
        if i > 30:
            break

        if torch.cuda.is_available():
            image = image.cuda()
        logit = model.forward(image.float()).cpu()
        image = torch.squeeze(image, dim=0)
        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.detach().numpy()
        idx = idx.numpy()

        predicted_labels.append(idx[0])
        predicted = str(predicted_labels)

        print("Target: " + 'placeholder' + " | Predicted: " + predicted)

        features = model.pretrained_model.features(image.float())
        # pooling_layer = nn.AdaptiveAvgPool2d(1)
        # pooled_features = pooling_layer(features)
        # pooled_features = pooled_features.view(pooled_features.size(0), -1)
        # flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        # max_index = torch.argmax(pooled_features, 0)

        features = features.cpu().detach().numpy()
        imagepack = image.cpu().detach().numpy()

        CAMs = np.zeros_like(imagepack)

        for index, scan_features in enumerate(features):
            cam = return_CAM(scan_features, weights, [idx[0]])
            img = imagepack[index]
            img = np.moveaxis(img, 0, -1)
            height, width, _ = img.shape
            resized = cv2.resize(cam[0], (width, height))
            # resized[resized <100] = 0

            #resized[np.logical_and(resized > 80, resized < 180)] = 0

            heatmap = cv2.applyColorMap(resized, cv2.COLORMAP_JET)

            result = heatmap * 0.5 + img * 0.5

            cv2.imwrite("image_{}_{}_{:.4}.png".format(i, index, probs[1] ), result)
            cv2.imwrite("image_{}_{}_orig.png".format(i, index), img)


def run(args):
    validation_dataset = MRDataset(args.task,
                                  args.plane, train=False, drop_researh_tags=args.drop_tags, researh_type=args.types,
                                  resize=True, dim=(256, 256), crop=True, transform=False, best_slice=best_slice_flag)

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=1, shuffle=-True, num_workers=0, drop_last=False)

    mrnet = model.MRNet2()
    mrnet.load_state_dict(torch.load(args.load_path))
    mrnet.eval()

    params = list(mrnet.parameters())
    weights = np.squeeze(params[-2].data.numpy())

    if torch.cuda.is_available():
        mrnet = mrnet.cuda()

    evaluate_model(mrnet, validation_loader, weights)

    print('-' * 30)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='bone marrow lesion',
                        choices=['cartialge lesion', 'bone marrow lesion', 'ill', 'synovitis'])
    parser.add_argument('-p', '--plane', type=str, default='cor',
                        choices=['sagittal', 'cor', 'axial'])
    parser.add_argument('--drop_tags', type=str, default=[ 'DIRTY'])
    parser.add_argument('--types', type=str, default=['FS', 'fs', 'STIR', 'SPAIR', 'DARK', ])
    parser.add_argument('--prefix_name', type=str, default='tipaFinal')


    parser.add_argument('--load_path', type=str,
                        default='models/model_Syn_AUG_toCAM_bone marrow lesion_cor_val_auc_0.6805_train_auc_0.7248_epoch_29.pth')
    args = parser.parse_args()
    return args

    #'models/model_abnormal_sagittal_8371.pth'
if __name__ == "__main__":
    model_type = 'MRnet2'
    transform_flag = False
    resize_flag = True
    best_slice_flag = True
    args = parse_arguments()
    run(args)
