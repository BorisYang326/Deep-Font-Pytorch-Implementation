import torch
from src.preprocess import TRANSFORMS_SQUEEZE
from PIL import Image
from src.model import FontResNet
import os
import torch.nn.functional as F
import argparse
from typing import List, Tuple
from src.model import SCAE, CNN, FontResNet

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_image_font_list(
    img_list_path: str, font_list_path: str
) -> Tuple[List[str], List[str]]:
    image_list = []
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


def get_font_name(output, font_books, gt_font_name):
    topk_values, topk_indices = torch.topk(output, dim=1, k=3)
    topk_indices_list = topk_indices.squeeze().tolist()
    topk_values_list = topk_values.squeeze().cpu().numpy()
    font_name_list = [font_books[i] for i in topk_indices_list]
    font_scores_list = [topk_values_list[idx] for idx in range(len(topk_indices_list))]
    hit_flag = gt_font_name in font_name_list
    return font_name_list, font_scores_list, hit_flag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, default="SCAE")
    parser.add_argument('num_classes', type=int, default=2383)
    parser.add_argument(
        'scae_weight_path', type=str, default="./weights/SCAE/scae_weights.pth"
    )
    parser.add_argument(
        'cnn_weight_path', type=str, default="./weights/CNN/cnn_weights_bk.pth"
    )
    parser.add_argument(
        'resent_weight_path',
        type=str,
        default="./weights/Resnet-50/ResNet_full_weights_13.pth",
    )
    parser.add_argument(
        'font_book_path', type=str, default="/public/dataset/AdobeVFR/fontlist.txt"
    )
    parser.add_argument('test_folder', type=str, default="./test_images/syn/train/")
    parser.add_argument('result_folder', type=str, default="./result/")
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
            gt_font_name = image_path.split('/')[-1].split('.')[0].split('_')[0]
            image = TRANSFORMS_SQUEEZE(Image.open(image_path))
            # image_crop = TRANSFORMS_CROP(Image.open(image_path))
            image = image.unsqueeze(0).to(DEVICE)
            # image_pil = transforms.ToPILImage()(image.squeeze(0).cpu())
            # image_crop_pil = transforms.ToPILImage()(image_crop)
            # image_pil.save(os.path.join(result_folder, image_path.split('/')[-1]))
            # image_crop_pil.save(os.path.join(result_folder, image_path.split('/')[-1].split('.')[0]+'_crop.jpg'))
            output = F.softmax(model(image), dim=1)
            font_name_list, font_scores_list, hit_flag = get_font_name(
                output, font_books, gt_font_name
            )
            print(
                "Test image {:d}/{:d}: {:s} -> [{:s}]".format(
                    k + 1, len(image_list), image_path, str(hit_flag)
                )
            )
            for font, score in zip(font_name_list, font_scores_list):
                print(f"Font: {font}, Score: {score}")
                print('-----------------------------------------------')


if __name__ == '__main__':
    main()
