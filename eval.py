import torch
from src.preprocess import TRANSFORMS_EVAL, TRANSFORMS_TRAIN,AUGMENTATION_LIST
from PIL import Image
import os
import torch.nn.functional as F
import argparse
from typing import List, Tuple
from torch import Tensor
from src.model import SCAE, CNN, FontResNet
import pickle
import matplotlib.pyplot as plt
from torchvision import transforms
from einops import rearrange
from src.utils import augment_hdf5_preprocess
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_image_font_list(
    img_list_path: str, font_list_path: str
) -> Tuple[List[str], List[str]]:
    image_list = []
    img_list_path = os.path.join(ROOT_DIR, img_list_path)
    for f in os.listdir(img_list_path):
        if os.path.isfile(os.path.join(img_list_path, f)) and (
            f.endswith('.jpg') or f.endswith('.png')
        ):
            image_list.append(os.path.join(img_list_path, f))

    with open(font_list_path, 'r') as f:
        font_books = f.readlines()
    font_books = [font_name.strip() for font_name in font_books]
    return image_list, font_books


def get_model(
    model_name: str,
    num_classes: int,
    scae_weight_path: str,
    cnn_weight_path: str,
    resent_weight_path: str,
) -> torch.nn.Module:
    if model_name == 'scae':
        model = SCAE().to(DEVICE)
        assert os.path.exists(scae_weight_path), 'No weights file for SCAE_model!'
        model._load_weights(scae_weight_path)
    elif model_name == 'cnn':
        model = CNN(num_classes, scae_weight_path).to(DEVICE)
        assert os.path.exists(cnn_weight_path), 'No weights file for CNN_model!'
        model._load_weights(cnn_weight_path)
    elif model_name in ['resnet18', 'resnet50']:
        model = FontResNet(num_classes, 'resnet50').to(DEVICE)
        assert os.path.exists(
            resent_weight_path
        ), 'No weights file for FontResNet_model!'
        model._load_weights(resent_weight_path)
    else:
        raise NotImplementedError
    return model


def get_font_name(output: Tensor, font_books: List[str], gt_font_name: str):
    topk_values, topk_indices = torch.topk(output, dim=1, k=3)
    topk_indices_list = topk_indices.squeeze().tolist()
    topk_values_list = topk_values.squeeze().cpu().numpy()
    font_name_list = [font_books[i] for i in topk_indices_list]
    font_scores_list = [topk_values_list[idx] for idx in range(len(topk_indices_list))]
    hit_flag = gt_font_name in font_name_list
    return font_name_list, font_scores_list, hit_flag


def draw_class_acc(class_acc_pkl_path: str):
    class_acc_dict = pickle.load(open(class_acc_pkl_path, 'rb'))
    plt.figure(figsize=(20, 10))
    plt.bar(range(len(class_acc_dict)), list(class_acc_dict.values()), align='center')
    plt.xticks(range(len(class_acc_dict)), list(class_acc_dict.keys()), rotation=90)
    plt.title('Class Accuracy')
    plt.savefig('class_acc.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="resnet50")
    parser.add_argument('--num_classes', type=int, default=2383)
    parser.add_argument(
        '--scae_weight_path', type=str, default="./weights/SCAE/scae_weights.pth"
    )
    parser.add_argument(
        '--cnn_weight_path', type=str, default="./weights/CNN/CNN_weights_39.pth"
    )
    parser.add_argument(
        '--resent_weight_path',
        type=str,
        default="./outputs/2023-10-22/21-13-49/saved_models/ResNet_weights_10.pth",
    )
    parser.add_argument(
        '--font_book_path', type=str, default="/public/dataset/AdobeVFR/fontlist.txt"
    )
    parser.add_argument('--test_folder', type=str, default="./test_images/syn/train/")
    parser.add_argument('--result_folder', type=str, default="./result/")
    args = parser.parse_args()
    if not os.path.isdir(args.result_folder):
        os.mkdir(args.result_folder)

    image_list, font_books = get_image_font_list(args.test_folder, args.font_book_path)
    model = get_model(
        args.model,
        args.num_classes,
        args.scae_weight_path,
        args.cnn_weight_path,
        args.resent_weight_path,
    )
    for k, image_path in enumerate(image_list):
        with torch.no_grad():
            model.eval()
            gt_font_name = image_path.split('/')[-1].split('.')[0].split('_')[0]
            image = TRANSFORMS_EVAL(Image.open(image_path))
            ### DEBUG PART ###
            # image_train = TRANSFORMS_TRAIN(Image.open(image_path))
            # image_train_pil = transforms.ToPILImage()(image_train)
            # image_train_pil.save(
            #     os.path.join(
            #         os.path.join(args.result_folder, 'train'),
            #         image_path.split('/')[-1].split('.')[0] + '_train.jpg',
            #     )
            # )

            # image_pil = transforms.ToPILImage()(image.squeeze(0).cpu())
            # image_pil.save(os.path.join(os.path.join(args.result_folder, 'eval'), image_path.split('/')[-1]))
            ### DEBUG PART END ###
            image = rearrange(image, 'c h w -> 1 c h w').to(DEVICE)
            output = F.softmax(model(image), dim=1)
            font_name_list, font_scores_list, hit_flag = get_font_name(
                output, font_books, gt_font_name
            )
            print(
                "Test image {:d}/{:d}: {:s} -> [{:s}]".format(
                    k + 1, len(image_list), image_path, str(hit_flag)
                )
            )
            print('Predicted label: {:d}'.format(torch.argmax(output, dim=1).item()))
            for font, score in zip(font_name_list, font_scores_list):
                print(f"Font: {font}, Score: {score}")
                print('-----------------------------------------------')


if __name__ == '__main__':
    # pkl_path = './multirun/2023-10-21/23-19-37/0/saved_models/class_accuracy.pkl'
    # pkl_path = './outputs/2023-10-21/16-32-10/saved_models/class_accuracy.pkl'
    # draw_class_acc(pkl_path)
    augment_hdf5_preprocess('/public/dataset/AdobeVFR/hdf5/VFR_syn_train.hdf5', AUGMENTATION_LIST, '/public/dataset/AdobeVFR/hdf5/VFR_syn_train_aug_bk.hdf5',4096)
    # main()
