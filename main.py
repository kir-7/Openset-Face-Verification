import torch

from train import train
from utils import create_dataloader, send_alert, read_json, write_json
from models.resnet import get_pretrained_resnet
from facenet_pytorch import MTCNN

import cv2
from PIL import Image

import os
import time


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

workers = 0 if os.name == 'nt' else 8 

class Pipeline:
    ''' A Pipeline is going to build the model and infer '''


    def __init__(self, model_path, detector=None, emb_f=None, thr_f=None, subjects=None, logger=None, mode='validate'):
        '''
        An Initializer for the pipeline
            Args:
                model_path: str : A path to a pytorch model that is trained to generate face embd
                detector : a model(mtcnn) that detect faces in imgs
                emb_f : str : a path to the file that stores the embedding vectors for all the images in the database
                thr_f : str : a path to the file that stores the threshold values for different classes of images
                subject : st]r : a path to the file that stores the labels to person data
                logger : an object that logs all the outputs of the validate function
                mode : whether to check for faces in the video stream or not
        '''
        self.model_path = model_path


        self.emb_f = emb_f
        self.thr_f = thr_f
        self.subjects = subjects

        self.logger = logger

        self.mode = mode

        self.model = None

        self.detector = detector

        self.cos_sim = lambda a, b: ((torch.dot(a, b)) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)) + 1)/(2) 


    def build_model(self):
        '''
        A function to build the model given the model path
        return: bool : whether the model is built or not
        '''
        if self.model_path is None:
            print("Model is not defined")
            return False

        if not os.path.isfile(self.model_path):
            print("model path incorrect")
            return False

        self.model = torch.load(self.model_path)
        return True

    def validate(self):
        '''
        This function is used to check the video get the faces, embdding vector from the video

        '''

        if self.detector is None:
            print("face detector not found!")
            return

        if self.model is None:
            if self.build_model() is False:
                print("Unable to load model ")
                return
            else:
                print("model loaded successfully")
        
        failure_faces = []

        # start streaming video from webcam
        video_capture = cv2.VideoCapture(0)
        thr_data = read_json(self.thr_f)
        base_thr = thr_data[list(thr_data.keys())[0]] 

        last_reset_time = time.time() 

        while True:
            ret, frame = video_capture.read()
            
            if not ret:
                print("error in capturing video")
                break

            opencv_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(opencv_image_rgb)

            # get face region coordinates
            faces = self.detector(pil_image)

            # get face bounding box for overlay
            if len(faces) > 0:
                for f in range(len(faces)):
                    extracted_img = faces[f]
                    extracted_img = extracted_img.unsqueeze(0)

                    extracted_img_emb  = self.model(extracted_img)

                    r = self.eval(extracted_img_emb[0])
                    if r[0] == 0:
                        for prev_failure in failure_faces:
                            if self.sim(prev_failure, r[2]) > base_thr:
                                trigger = True

                        if not trigger:
                            self.handle_failure(r)
                            failure_faces.append(r[2])
    
                    else:
                        self.handle_success(r)

            if time.time() - last_reset_time >= 600:  # 600 seconds = 10 minutes
                failure_faces.clear()
                last_reset_time = time.time()
    
    def sim(a, b):
        """Takes 2 vectors a, b and returns the cosine similarity according
            to the definition of the dot product"""
        dot_product = torch.dot(a, b)
        norm_a = torch.linalg.vector_norm(a)
        norm_b = torch.linalg.vector_norm(b)
        return (dot_product / (norm_a * norm_b) + 1)/2


    def eval(self, emb):
        '''
            This function is used to check whether face in the database or not

            Args: emb : embedding vector of the detected face

            Returns: a tuple (v, l, e, th)
                v : 1 if found 0 if not
                l : None is v =0 else label of found face
                e : the embeding vector
                th : the threshold value of the label it matched with
                    th = None is v = 0 else 0<th<1

        '''
        r = (0, None, emb, None)

        if self.emb_f is None:
            return r
        
        if self.thr_f is None:
            return r    


        emb_data = read_json(self.emb_f)
        thr_data = read_json(self.thr_f)
        #  emb_f has data in format {label:emb_v}
        # thr_f has data in format {label:threshold}

        for label, vect in emb_data.items():
            if self.cos_sim(vect, emb) >= thr_data[label]:
                r = (1, label, emb, thr_data[label])
                return r

        return r


    def handle_success(self, r):
        '''
        A function that is handles the condition when we found a match
        args: r (v, l, e, t): v = 1
                              l = label of match
                              e = emb vector
                              t = threshold
        '''
        print("found a match logging")
        self.log_to_logger(r)

        return

    def handle_failure(self, r):
        '''A function that handles the condition when no match found 
          Will send alert after we print the failure statement and log it
          args: r (v, l, e, t): v = 1
                              l = label of match
                              e = emb vector
                              t = threshold  
        '''
        print("A new Face found")
        self.log_to_logger(r)

        return

    def log_to_logger(self, status):
        '''
        A function that logs all the matches and misses
        '''
        if self.logger is None:
            print('no logger found cant log')
            return

        if status[0] == 0:
            data = {str(time.time()): 'An intruder found alerted!'}
        else:
            data = {str(time.time()): f'Match found label : {status[1]}'}

        write_json(self.logger, data)

        return


if __name__ == '__main__':

    to_train = False

    if to_train:

        data_dir = ''
        batch_size = 32
        epochs = 8
        emb_dim = 512

        n_classes, train_loader, val_loader = create_dataloader(data_dir, batch_size, workers, device)

        resnet = get_pretrained_resnet("vggface2")

        train(resnet, train_loader, val_loader, n_classes, emb_dim, epochs, device, save_model="trained_models/resnet_indian_faces_8epochs")

    else:
        model_path = "trained_models\indian face resnet v3"

        detector =  MTCNN(image_size=160, margin=0, min_face_size=20,
                        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                        device=device
                    )
        
        emb_f = "files\emb_f.json"
        thr_f = "files\thr_f.json"
        logger = "files\logger.json"

        p = Pipeline(model_path, detector, emb_f=emb_f, thr_f=thr_f, logger=logger)
        
        p.validate()
