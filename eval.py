import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from preprocess import TRANSFORMS_SQUEEZE,TRANSFORMS_CROP
from PIL import Image
from model import SCAE,CNN,FontResNet
import os

scae_weights_path = './weights/scae_weights.pth'
cnn_weights_path = './weights/cnn_weights_bk.pth'
resent_weights_path = './weights/resnet_weights_0.pth'
test_folder = './test_images/gen/'
result_folder = './result/'


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval(model, data, weights_path, batch_size=32):
    model.load_weights(weights_path)
    return model.evaluate(data, batch_size=batch_size)


if __name__ == '__main__':
    """ For test images in a folder """
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    SCAE_model = SCAE().to(DEVICE)
    assert os.path.exists(scae_weights_path),'No weights file for SCAE_model!'
    SCAE_model.load_weights(scae_weights_path)
    model = CNN(SCAE_model.encoder).to(DEVICE)
    assert os.path.exists(cnn_weights_path),'No weights file for CNN_model!'
    model.load_weights(cnn_weights_path)
    # model = FontResNet().to(DEVICE)
    # assert os.path.exists(resent_weights_path),'No weights file for FontResNet_model!'
    # model.load_weights(resent_weights_path)
    image_list = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f)) and (f.endswith('.jpg') or f.endswith('.png'))]

    with open('fontbooks.txt', 'r') as f:
        font_books = f.readlines()
    font_books = [font_name.strip() for font_name in font_books]

    for k, image_path in enumerate(image_list):
        with torch.no_grad():
            gt_font_name = image_path.split('/')[-1].split('.')[0].split('_')[1]
            image = TRANSFORMS_SQUEEZE(Image.open(image_path))
            # image_crop = TRANSFORMS_CROP(Image.open(image_path))
            image = image.unsqueeze(0).to(DEVICE)
            image_pil = transforms.ToPILImage()(image.squeeze(0).cpu())
            # image_crop_pil = transforms.ToPILImage()(image_crop)
            image_pil.save(os.path.join(result_folder, image_path.split('/')[-1]))
            # image_crop_pil.save(os.path.join(result_folder, image_path.split('/')[-1].split('.')[0]+'_crop.jpg'))
            output = model(image)
            topk_values, topk_indices = torch.topk(output, k=3)
            topk_indices_list = topk_indices.squeeze().tolist()
            topk_values_list = topk_values.squeeze().cpu().numpy()
            font_name_list = [font_books[i] for i in topk_indices_list]
            font_scores_list = [topk_values_list[idx] for idx in range(len(topk_indices_list))]
            hit_flag = gt_font_name in font_name_list
            print("Test image {:d}/{:d}: {:s} -> [{:s}]".format(k+1, len(image_list), image_path, str(hit_flag)))
            for font, score in zip(font_name_list, font_scores_list):
                print(f"Font: {font}, Score: {score}")
            print('-----------------------------------------------')