import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import FontResNet
from dataset import VFRRealUDataset, VFRSynDataset
from trainer import FontClsTrainer
from torch.utils.data import ConcatDataset
from preprocess import TRANSFORMS_SQUEEZE

VFR_real_u_path = (
    '/public/dataset/AdobeVFR/AdobeVFR/Raw Image/VFR_real_u/scrape-wtf-new'
)
VFR_syn_train_path = '/public/dataset/AdobeVFR/AdobeVFR/Raw Image/VFR_syn_train'
VFR_syn_val_path = '/public/dataset/AdobeVFR/AdobeVFR/Raw Image/VFR_syn_val'
VFR_syn_font_list_path = '/public/dataset/AdobeVFR/AdobeVFR/font_list.txt'
scae_weights_path = './weights/scae_weights.pth'
cls_weights_path = './weights/'


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4096 * 2
EPOCHS = 30
NUMBER_OF_WORKERS = 16
LEARNING_RATE = 1e-4
PREFETCH_FACTOR = 2


if __name__ == '__main__':
    ## Dataset ##
    scae_real_dataset = VFRRealUDataset(
        root_dir=VFR_real_u_path, transform=TRANSFORMS_SQUEEZE
    )
    scae_syn_dataset = VFRSynDataset(
        root_dir=VFR_syn_train_path,
        font_list_path=VFR_syn_font_list_path,
    )

    combined_scae_dataset = ConcatDataset([scae_real_dataset, scae_syn_dataset])
    supervised_train_dataset = VFRSynDataset(
        root_dir=VFR_syn_train_path,
        transform=TRANSFORMS_SQUEEZE,
    )
    supervised_eval_dataset = VFRSynDataset(
        root_dir=VFR_syn_val_path,
        transform=TRANSFORMS_SQUEEZE,
    )
    ## Data Loader ##
    scae_loader = DataLoader(
        combined_scae_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUMBER_OF_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR,
    )
    supervised_train_loader = DataLoader(
        supervised_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUMBER_OF_WORKERS,
        pin_memory=False,
        prefetch_factor=PREFETCH_FACTOR,
    )
    supervised_test_loader = DataLoader(
        supervised_eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUMBER_OF_WORKERS,
        pin_memory=False,
        prefetch_factor=PREFETCH_FACTOR,
    )
    ## Check Dataloader ##
    # scae_sample = next(iter(scae_loader))
    # scae_sample_pil = transforms.ToPILImage()(scae_sample[0][0])
    # scae_sample_pil.save('scae_sample.png')
    # cnn_train_sample = next(iter(cnn_train_loader))
    # cnn_train_sample_pil = transforms.ToPILImage()(cnn_train_sample[0][0])
    # cnn_train_sample_pil.save('cnn_train_sample.png')
    # cnn_eval_sample = next(iter(cnn_train_loader))
    # cnn_eval_sample_pil = transforms.ToPILImage()(cnn_eval_sample[0][0])
    # cnn_eval_sample_pil.save('cnn_eval_sample.png')
    ### SCAE Part ###
    # SCAE_model = SCAE().to(DEVICE)
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs for SCAE model!")
    #     SCAE_model = nn.DataParallel(SCAE_model)
    # if os.path.exists(scae_weights_path):
    #     SCAE_model.load_state_dict(torch.load(scae_weights_path))
    # else:
    #     optimizer_scae = optim.AdamW(SCAE_model.parameters(), lr=LEARNING_RATE)
    #     mse_loss = nn.MSELoss()
    #     scae_trainer = SCAETrainer(SCAE_model,optimizer_scae,mse_loss,DEVICE,scae_weights_path)
    #     scae_trainer._train(scae_loader,EPOCHS)
    #     scae_trainer._save_weights()
    ### CNN Part ###
    # if torch.cuda.device_count() > 1:
    #     CNN_model = CNN(SCAE_model.module.encoder,finetune_ratio_Cu=1e-5)
    #     cnn_optimizer = CNN_model.get_optimizer(LEARNING_RATE)
    #     print(f"Using {torch.cuda.device_count()} GPUs for CNN model!")
    #     CNN_model = nn.DataParallel(CNN_model)
    # else:
    #     CNN_model = CNN(SCAE_model.module.encoder,finetune_ratio_Cu=1e-5)
    #     cnn_optimizer = CNN_model.get_optimizer(LEARNING_RATE)
    # celoss = nn.CrossEntropyLoss()
    # cnn_trainer = CNNTrainer(CNN_model,cnn_optimizer,celoss,DEVICE,cnn_test_loader,cnn_weights_path)
    # cnn_trainer._train(cnn_train_loader,EPOCHS)
    # cnn_trainer.writer.close()
    ### ResNet Part ###
    resnet_model = FontResNet()
    celoss = nn.CrossEntropyLoss()
    resnet_optimizer = resnet_model.get_optimizer(LEARNING_RATE)
    if torch.cuda.device_count() > 1:
        resnet_model = nn.DataParallel(resnet_model)
    else:
        resnet_model = resnet_model
    resnet_trainer = FontClsTrainer(
        resnet_model,
        resnet_optimizer,
        celoss,
        DEVICE,
        supervised_test_loader,
        cls_weights_path,
    )
    resnet_trainer._train(supervised_train_loader, EPOCHS)
    resnet_trainer.writer.close()
