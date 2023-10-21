import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from src.preprocess import TRANSFORMS_SQUEEZE,TRANSFORMS_CROP
from PIL import Image
from src.model import SCAE,CNN,FontResNet
import os
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scae_weights_path = os.path.join(ROOT_DIR,'./weights/scae_weights.pth')
cnn_weights_path = os.path.join(ROOT_DIR,'./weights/cnn_weights_bk.pth')
resent_weights_path = os.path.join(ROOT_DIR,'./weights/Resnet-50/ResNet_full_weights_13.pth')
test_folder = os.path.join(ROOT_DIR,'./test_images/syn/train/')
result_folder = os.path.join(ROOT_DIR,'./result/')
# font_book_path = os.path.join(ROOT_DIR,'./test_images/fontlist_20.txt')
font_book_path = '/public/dataset/AdobeVFR/fontlist.txt'


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval(model, data, weights_path, batch_size=32):
    model.load_weights(weights_path)
    return model.evaluate(data, batch_size=batch_size)


if __name__ == '__main__':
    """ For test images in a folder """
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    # SCAE_model = SCAE().to(DEVICE)
    # assert os.path.exists(scae_weights_path),'No weights file for SCAE_model!'
    # SCAE_model.load_weights(scae_weights_path)
    # model = CNN(SCAE_model.encoder).to(DEVICE)
    # assert os.path.exists(cnn_weights_path),'No weights file for CNN_model!'
    # model.load_weights(cnn_weights_path)
    model = FontResNet(2383,'resnet50').to(DEVICE)
    assert os.path.exists(resent_weights_path),'No weights file for FontResNet_model!'
    model._load_weights(resent_weights_path)
    image_list = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f)) and (f.endswith('.jpg') or f.endswith('.png'))]

    with open(font_book_path, 'r') as f:
        font_books = f.readlines()
    font_books = [font_name.strip() for font_name in font_books]

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
            output = F.softmax(model(image),dim=1)
            topk_values, topk_indices = torch.topk(output,dim=1, k=3)
            topk_indices_list = topk_indices.squeeze().tolist()
            topk_values_list = topk_values.squeeze().cpu().numpy()
            font_name_list = [font_books[i] for i in topk_indices_list]
            font_scores_list = [topk_values_list[idx] for idx in range(len(topk_indices_list))]
            hit_flag = gt_font_name in font_name_list
            print("Test image {:d}/{:d}: {:s} -> [{:s}]".format(k+1, len(image_list), image_path, str(hit_flag)))
            for font, score in zip(font_name_list, font_scores_list):
                print(f"Font: {font}, Score: {score}")
            print('-----------------------------------------------')