import os
from datetime import datetime
import numpy as np
import threading

from alive_progress import alive_bar
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms as T
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.cuda.amp import GradScaler, autocast
from PIL import Image

from .generator import Generator,Discriminator
from .config import ConfigManager
from .logger import Logger
from .messages import display_goodbye_message

class cgan:

    def __init__(self,**kwargs):
        self.running = True
        self.save_requested = False

        self.config=ConfigManager(kwargs)
        
        self.config.load_or_save()
        self.logger=Logger(self.config.log_file,stdout=True)

        self.start_epoch = 0
        self.optimizer_gen = None
        self.optimizer_dis = None
        self.data_loader = None
        self.image_channel = None
        self.char_height = None
        self.char_width = None
        self.z_dim = None
        self.num_classes_data = None
        self.fc_neuron = None
        self.gen_model = None
        self.dis_model = None
        self.loss_fn = None
        self.scaler = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.configure()

  

    def print_info(self):
        self.logger.log("Training Configuration:")
        self.logger.log(f"  Output Directory:         {self.config.output_dir}")
        self.logger.log(f"  Model Directory:          {self.config.model_dir}")
        self.logger.log(f"  Image Directory:          {self.config.image_dir}")
        self.logger.log(f"  Save Checkpoints:         {'Yes' if self.config.save_checkpoints else 'No'}")
        
        if self.config.save_checkpoints:
            self.logger.log(f"  Checkpoint Interval:      {self.config.checkpoint_interval}")
        self.logger.log(f"  Optimizer:                {self.config.optimizer}")
        self.logger.log(f"  Gen LR:                   {self.config.gen_lr}")
        self.logger.log(f"  Dis LR:                   {self.config.dis_lr}")

        if self.config.gen_model_path is not None and self.config.dis_model_path is not None: 
            self.logger.log(f"  Generator Model Path:     {self.config.gen_model_path if self.config.gen_model_path else 'Not provided'}")
            self.logger.log(f"  Discriminator Model Path: {self.config.dis_model_path if self.config.dis_model_path else 'Not provided'}")
            self.logger.log(f"  Starting Epoch:           {self.start_epoch}")
            
        self.logger.log(f"  Epochs:                   {self.config.epochs}")


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
        

    def configure(self):
        # Transform
        transform_data = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5)
        ])

        # Load MNIST dataset
        train_dataset = MNIST(root='', train=True, transform=transform_data, download=True)
        valid_dataset = MNIST(root='', train=False, transform=transform_data, download=True)

        # Concat dataset
        dataset = ConcatDataset([train_dataset, valid_dataset])

        # DataLoader
        data_batch_size = 100
        self.data_loader = DataLoader(dataset, batch_size=data_batch_size, shuffle=True, num_workers=4, pin_memory=True)

        x, _ = next(iter(self.data_loader))
        self.image_channel= x.shape[1] 
        self.char_height = x.shape[2]
        self.char_width = x.shape[3]
        self.z_dim = self.char_height
        self.num_classes_data = 10
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
                self.load_checkpoint(self.gen_model, self.optimizer_gen, self.config.gen_model_path)
                self.load_checkpoint(self.dis_model, self.optimizer_dis, self.config.dis_model_path)
            except Exception as ex:
                self.logger.log(ex)
                self.logger.log("Error loading models")
                exit(code=1)
        elif self.config.gen_model_path is not None or self.config.dis_model_path is not None: 
            self.start_epoch = 0
            self.logger.log("You need to give both model paths to load a model set")



        self.loss_fn = nn.BCEWithLogitsLoss().to(self.device)  
        self.scaler = GradScaler()


    def train_one_epoch(self,dis_model, gen_model, train_loader, loss_fn, optimizer_dis, optimizer_gen, scaler):
        dis_model.train()
        gen_model.train()
        
        d_loss_total = 0.0
        g_loss_total = 0.0
        num_batches = 0    
        total=len(train_loader)
        with alive_bar(total) as bar:

            for x_real, y in train_loader:
                batch_size = x_real.shape[0]
                x_real = x_real.flatten(1).to(self.device)
                y_real = torch.ones(batch_size, 1, device=self.device)
                one_hot = torch.nn.functional.one_hot(y, self.num_classes_data).to(self.device)
                concat_data_dis_real = torch.cat((one_hot, x_real), dim=1)

                noise = torch.randn(batch_size, self.z_dim, device=self.device)
                concat_data_gen_fake = torch.cat((one_hot, noise), dim=1)
                
                optimizer_dis.zero_grad()
                optimizer_gen.zero_grad()
                
                with autocast():
                    # Discriminator real loss
                    out_real = dis_model(concat_data_dis_real)
                    loss_real = loss_fn(out_real, y_real)
                    
                    # Discriminator fake loss
                    x_fake = gen_model(concat_data_gen_fake)
                    concat_data_dis_fake = torch.cat((one_hot, x_fake), dim=1)
                    y_fake = torch.zeros(batch_size, 1, device=self.device)
                    out_fake = dis_model(concat_data_dis_fake)
                    loss_fake = loss_fn(out_fake, y_fake)

                    loss_dis = loss_real + loss_fake
                
                scaler.scale(loss_dis).backward()
                torch.nn.utils.clip_grad_norm_(dis_model.parameters(), max_norm=1)

                scaler.step(optimizer_dis)
                scaler.update()
                
                optimizer_dis.zero_grad()
                
                # Generator loss
                with autocast():
                    noise = torch.randn(batch_size, self.z_dim, device=self.device)
                    concat_data_gen = torch.cat((one_hot, noise), dim=1)
                    x_gen = gen_model(concat_data_gen)
                    concat_data_dis_gen = torch.cat((one_hot, x_gen), dim=1)
                    y_gen = torch.ones(batch_size, 1, device=self.device)
                    out_gen = dis_model(concat_data_dis_gen)
                    loss_gen = loss_fn(out_gen, y_gen)
                
                scaler.scale(loss_gen).backward()
                torch.nn.utils.clip_grad_norm_(gen_model.parameters(), max_norm=1)

                scaler.step(optimizer_gen)
                scaler.update()
                
                optimizer_gen.zero_grad()

                d_loss_total += loss_dis.item()
                g_loss_total += loss_gen.item()
                num_batches += 1
                bar.text(f"Epoch: {self.current_epoch}:{self.config.epochs} Gen/Dis Loss:{(g_loss_total/num_batches):.4f}/{(d_loss_total/num_batches):.4f}: ")
                bar()

        return {'d_loss':d_loss_total/num_batches,'g_loss':g_loss_total/num_batches}


    def log_metrics(self, epoch, d_loss, g_loss):
        self.logger.log(f"Epoch: {epoch}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}",stdout=None)

    def save_checkpoint(self, filename):
        self.logger.log(f"Saving epoch {self.current_epoch} Checkpoint: {filename}")
        torch.save({
            'epoch': self.current_epoch,
            'gen_model_state_dict': self.gen_model.state_dict(),
            'gen_optimizer_state_dict': self.optimizer_gen.state_dict(),
            'dis_model_state_dict': self.dis_model.state_dict(),
            'dis_optimizer_state_dict': self.optimizer_dis.state_dict(),
            'optimizer': self.config.optimizer  
        }, filename)

    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def load_checkpoint(self,filename, gen_model, optimizer_gen, dis_model, optimizer_dis, config, logger=None):
        if logger:
            logger.log(f"Loading checkpoint from: {filename}")

        checkpoint = torch.load(filename)

        gen_model.load_state_dict(checkpoint['gen_model_state_dict'])
        optimizer_gen.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        dis_model.load_state_dict(checkpoint['dis_model_state_dict'])
        optimizer_dis.load_state_dict(checkpoint['dis_optimizer_state_dict'])
        config.optimizer = checkpoint['optimizer']
        
        self.curent_epoch=checkpoint['epoch']

    def save_training_image(self,epoch,epochs):
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
        img_grid= make_grid(outputs, 3)
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

    def save(self):
        self.logger.log("Saving Checkpopint")
        self.save_checkpoint( os.path.join(self.config.model_dir, f"cgan_epoch_{self.current_epoch}.pth"))

    # Training loop
    def train(self):
        for epoch in range(self.start_epoch, self.config.epochs):
            if self.running!=True:
                return
            if self.save_requested:
                self.save_requested=None
                self.save()
            elif self.config.save_checkpoints and \
                 (epoch % self.config.checkpoint_interval == 0 or epoch == self.config.epochs - 1):
                self.save()

            self.current_epoch = epoch
            results=self.train_one_epoch(self.dis_model, self.gen_model, self.data_loader, self.loss_fn, self.optimizer_dis, self.optimizer_gen, self.scaler)
            
            # log metrics
            self.log_metrics(epoch,results['d_loss'], results['g_loss'])
            self.save_training_image(epoch, self.config.epochs)
        self.running=False

    def start(self):
        work_thread = threading.Thread(target=self.train)
        work_thread.start()

        return work_thread

    def stop(self):
        self.running = False

    def queue_save(self):
        self.save_requested = True
            
