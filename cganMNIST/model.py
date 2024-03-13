import os
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision import transforms as T
from torchvision.datasets import EMNIST
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image
import numpy as np

from .generator import Generator,Discriminator


class model:
    def __init__(self,config,logger,device):
        self.device=device
        self.logger=logger
        self.config=config
        self.optimizer_gen = None
        self.optimizer_dis = None
        self.data_loader = None
        self.image_channel = None
        self.char_height = None
        self.char_width = None
        self.z_dim = None
        self.fc_neuron = None
        self.model = None
        self.loss_fn = None
        self.scaler = None

        self.configure()

    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
 
    def configure(self):
        # Transform
        #transform_data = T.Compose([
        #    T.ToTensor(),
        #    T.Normalize(mean=0.5, std=0.5)
        #])
 
        transform_data=T.Compose([
            lambda img: T.functional.rotate(img, -90),
            lambda img: T.functional.hflip(img),
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5)
        ])        

        # Load EMNIST dataset
        train_dataset = EMNIST(root='',split='byclass', train=True , transform=transform_data, download=True)
        valid_dataset = EMNIST(root='',split='byclass', train=False, transform=transform_data, download=True)

        # Concat dataset
        dataset = ConcatDataset([train_dataset, valid_dataset])

        # DataLoader
        self.data_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, 
                                      num_workers=self.config.workers, pin_memory=True)

        x, _ = next(iter(self.data_loader))
        self.image_channel= x.shape[1] 
        self.char_height = x.shape[2]
        self.char_width = x.shape[3]
        self.z_dim = self.char_height
        self.num_classes_data = 62
        self.fc_neuron = 256

        self.logger.log(f'Height: {self.char_height}')
        self.logger.log(f'Width: {self.char_width}')
        self.logger.log(f'Neuron: {self.fc_neuron}')

        # Initialize models
        self.gen_model = Generator(self.char_height,self.char_width,self.fc_neuron).to(self.device)
        self.dis_model = Discriminator(self.fc_neuron).to(self.device)

        self.gen_model.apply(self.weights_init)
        self.dis_model.apply(self.weights_init)

        
        self.configure_optimizers()
        
        # Load pre-trained models if paths are provided, but only if both paths are provided
        if self.config.gen_model_path is not None and self.config.dis_model_path is not None: 
            self.logger.log("Loading Models")
            try:
                self.config.start_epoch=self.load_checkpoint(self.config.model_path)
            except Exception as ex:
                self.logger.log(ex)
                self.logger.log("Error loading models")
                exit(code=1)
        elif self.config.gen_model_path is not None or self.config.dis_model_path is not None: 
            self.start_epoch = 0
            self.logger.log("You need to give both model paths to load a model set")

        self.loss_fn = nn.BCEWithLogitsLoss().to(self.device)  
        self.scaler = GradScaler()


    def configure_optimizers(self):
        if self.config.optimizer == 'adam':
            self.optimizer_gen = torch.optim.Adam(self.gen_model.parameters(), lr=self.config.gen_lr)
            self.optimizer_dis = torch.optim.Adam(self.dis_model.parameters(), lr=self.config.dis_lr)
        elif self.config.optimizer == 'rmsprop':
            self.optimizer_gen = torch.optim.RMSprop(self.gen_model.parameters(), lr=self.config.gen_lr)
            self.optimizer_dis = torch.optim.RMSprop(self.dis_model.parameters(), lr=self.config.dis_lr)
        elif self.config.optimizer == 'sgd':
            self.optimizer_gen = torch.optim.SGD(self.gen_model.parameters(), lr=self.config.gen_lr, momentum=0.9)
            self.optimizer_dis = torch.optim.SGD(self.dis_model.parameters(), lr=self.config.dis_lr, momentum=0.9)
    
    def save(self, filename):
        self.logger.log(f"Saving epoch {self.current_epoch} Checkpoint: {filename}")
        torch.save({
            'epoch': self.current_epoch,
            'gen_model_state_dict': self.gen_model.state_dict(),
            'gen_optimizer_state_dict': self.optimizer_gen.state_dict(),
            'dis_model_state_dict': self.dis_model.state_dict(),
            'dis_optimizer_state_dict': self.optimizer_dis.state_dict(),
            'optimizer': self.config.optimizer  
        }, filename)

    def load(self,filename, logger=None):
        if logger:
            logger.log(f"Loading checkpoint from: {filename}")

        checkpoint = torch.load(filename)

        self.gen_model.load_state_dict(checkpoint['gen_model_state_dict'])
        self.optimizer_gen.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        self.dis_model.load_state_dict(checkpoint['dis_model_state_dict'])
        self.optimizer_dis.load_state_dict(checkpoint['dis_optimizer_state_dict'])
        self.config.optimizer = checkpoint['optimizer']
        return  checkpoint['epoch']

    def save_training_image(self,epoch):
        # Ensure the save directory exists
        os.makedirs(self.config.image_dir, exist_ok=True)

        # creating images of numbers one to ten
        target_label= torch.tensor(range(self.num_classes_data))
        outputs= torch.zeros(2, self.image_channel, self.char_height, self.char_width,device=self.device)
        for number in target_label:
            with torch.no_grad():
                bs= 1
                z_test= torch.randn(bs, self.z_dim).to(self.device)
                target_onehot= torch.nn.functional.one_hot(number, self.num_classes_data).to(self.device)
                target_onehot= torch.unsqueeze(target_onehot, dim=0)
                concat_target_data= torch.cat((target_onehot, z_test), dim=1)
                output= torch.unsqueeze(self.gen_model(concat_target_data), dim=0)
                outputs= torch.cat((output.view(bs,self.image_channel, self.char_height,self.char_width), outputs), dim=0)

        # display the created images
        img_grid= make_grid(outputs, 8)
        # Convert the tensor to [0, 255] and change order of dimensions to HxWxC
        img = img_grid.cpu().detach().numpy()
        img = (np.transpose(img, (1, 2, 0)) * 255).astype(np.uint8)

        # Convert numpy array to PIL Image
        img = Image.fromarray(img)

        # Save the image with filename based on the epoch
        filename = os.path.join(self.config.image_dir, f"img_{epoch}.png")
        img.save(filename)

        # Append the filename to a text file
        with open(os.path.join(self.config.output_dir, 'images.txt'), 'a') as f:
            f.write(filename + '\n')        

    