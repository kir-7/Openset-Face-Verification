
import torch

from train import train
from utils import create_dataloader
from models.resnet import get_pretrained_resnet
import os


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

workers = 0 if os.name == 'nt' else 8 


if __name__ == '__main__':
    data_dir = '0'
    batch_size = 32
    epochs = 8
    emb_dim = 512

    n_classes, train_loader, val_loader = create_dataloader(data_dir, batch_size, workers)

    resnet = get_pretrained_resnet("vggface2")

    train(resnet, train_loader, val_loader, n_classes, emb_dim, epochs, device, save_model="trained_models/resnet_indian_faces_8epochs")


    '''
        Here the code for loading the trained model after finetuning and
        loading the embds.json file that contains the embds for allowed images and 
        capturing the video and using mtcnn to get the face and calculating the embds for that 
        and comparing the generated embds to embds in database and send alert based on that.

        This project is still in progress and will be updated after finishing the remaining part   
    '''