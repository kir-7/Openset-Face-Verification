import torch

from facenet_pytorch import MTCNN, fixed_image_standardization
from facenet_pytorch.models.utils import training
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

import numpy as np
from collections import defaultdict
import os

from main import device


def create_copped_data(data_dir, dest_dir, workers, image_size=160, batch_size=32):
    
    mtcnn = MTCNN(image_size=image_size, 
                  margin=0, 
                  min_face_size = 20,
                  threshold=[0.6, 0.7, 0.7],
                  factor=0.709,
                  post_process=True,
                  device=device)

    dataset = datasets.ImageLoader(data_dir,
                                   transform=transforms.Resize(512, 512))
    

    loader = DataLoader(dataset,
                        num_workers=workers,
                        batch_size=batch_size,
                        collate_fn=training.collate_pil)
    
    images_in_labels = defaultdict(int)

    print("Cropping Data")

    for i, (x, y) in enumerate(loader):
        for j in range(min(batch_size, len(y))):
            path = dest_dir + f"/{y[j]}/{images_in_labels[y[j]]}.jpg"
            if not os.path.exists(path):
                images_in_labels[y[j]] += 1
                mtcnn(x[j], save_path=path)

        print('\rBatch {} of {} '.format(i + 1, len(loader)), end='')

    return


def create_dataloader(data_dir, batch_size, workers):


    trans = transforms.Compose([np.float32,
                                transforms.ToTensor(),
                                fixed_image_standardization
                                ])

    dataset = datasets.ImageFolder(data_dir , transform=trans)
    img_inds = np.arange(len(dataset))
    np.random.shuffle(img_inds)
    train_inds = img_inds[:int(0.8 * len(img_inds))]
    val_inds = img_inds[int(0.8 * len(img_inds)):]

    classes_cropped = os.listdir(data_dir)

    n_classes = len(classes_cropped)

    print(f"Found {n_classes} classes in the cropped folder")

    train_loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_inds)
    )
    val_loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_inds)
    )

    return n_classes, train_loader, val_loader 

