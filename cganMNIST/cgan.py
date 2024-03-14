import os
from datetime import datetime
import threading

from alive_progress import alive_bar
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms as T
from torch.cuda.amp import GradScaler, autocast

from .config import ConfigManager
from .logger import Logger
from .model import model

class cgan:

    def __init__(self,**kwargs):
        self.running = True
        self.save_requested = False
        self.start_epoch = 0

        self.config=ConfigManager(kwargs)
        self.config.load_or_save()
        self.logger=Logger(self.config.log_file,stdout=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model=model(self.config,self.logger,device=self.device)


    # THREADING
    def start(self):
        work_thread = threading.Thread(target=self.train)
        work_thread.start()
        return work_thread

    def stop(self):
        self.running = False

    def queue_save(self):
        self.save_requested = True
            
    def save(self):
        self.logger.log("Saving Checkpopint")
        self.model.save( os.path.join(self.config.model_dir, f"cgan_epoch_{self.current_epoch}.pth"))

    def print_info(self):
        self.logger.log("Training Configuration:")
        self.logger.log(f"  Model Type :              {self.config.model_type}")
        self.logger.log(f"  Output Directory:         {self.config.output_dir}")
        self.logger.log(f"  Model Directory:          {self.config.model_dir}")
        self.logger.log(f"  Image Directory:          {self.config.image_dir}")
        self.logger.log(f"  Save Checkpoints:         {'Yes' if self.config.save_checkpoints else 'No'}")
        self.logger.log(f"  Workers:                  {self.config.workers}")
        self.logger.log(f"  Batch:                    {self.config.batch_size}")
        
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
    
    # Training loop
    def train(self):
        start_time = time.time()  # Record start time
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
            results=self.train_one_epoch()
            
            # log metrics
            self.logger.log(f"Epoch: {epoch}, D Loss: {results['d_loss']:.4f}, G Loss: {results['g_loss']:.4f}, Time: {results['time']}",stdout=None)            
            self.model.save_training_image(epoch)
        self.running=False
        end_time = time.time()  # Record end time after training completes
        total_time = end_time - start_time  # Calculate total execution time
        avg_time = total_time/self.current_epoch
        self.logger.log(f"Total Epocs: {self.current_epoch}, Total Time: {total_time}, Average Time {avg_time}",stdout=None)

    # Single training instance
    def train_one_epoch(self):
        start_time = time.time()  # Record start time
        self.model.dis_model.train()
        self.model.gen_model.train()
        
        d_loss_total = 0.0
        g_loss_total = 0.0
        num_batches = 0    
        total=len(self.model.data_loader)
        with alive_bar(total) as bar:

            for x_real, y in self.model.data_loader:
                if self.running!=True:
                    return {'d_loss':0,'g_loss':0, 'time':0}
                
                batch_size = x_real.shape[0]
                x_real = x_real.flatten(1).to(self.device)
                y_real = torch.ones(batch_size, 1, device=self.device)
                one_hot = torch.nn.functional.one_hot(y, self.model.num_classes_data).to(self.device)
                concat_data_dis_real = torch.cat((one_hot, x_real), dim=1)

                noise = torch.randn(batch_size, self.model.z_dim, device=self.device)
                concat_data_gen_fake = torch.cat((one_hot, noise), dim=1)
                
                self.model.optimizer_dis.zero_grad()
                self.model.optimizer_gen.zero_grad()
                
                with autocast():
                    # Discriminator real loss
                    out_real = self.model.dis_model(concat_data_dis_real)
                    loss_real = self.model.loss_fn(out_real, y_real)
                    
                    x_fake = self.model.gen_model(concat_data_gen_fake)
                    concat_data_dis_fake = torch.cat((one_hot, x_fake), dim=1)
                    y_fake = torch.zeros(batch_size, 1, device=self.device)
                    out_fake = self.model.dis_model(concat_data_dis_fake)
                    loss_fake = self.model.loss_fn(out_fake, y_fake)

                    loss_dis = loss_real + loss_fake
                
                self.model.scaler.scale(loss_dis).backward()
                torch.nn.utils.clip_grad_norm_(self.model.dis_model.parameters(), max_norm=1)

                self.model.scaler.step(self.model.optimizer_dis)
                self.model.scaler.update()
                
                self.model.optimizer_dis.zero_grad()
                
                with autocast():
                    noise = torch.randn(batch_size, self.model.z_dim, device=self.device)
                    concat_data_gen = torch.cat((one_hot, noise), dim=1)
                    x_gen = self.model.gen_model(concat_data_gen)
                    concat_data_dis_gen = torch.cat((one_hot, x_gen), dim=1)
                    y_gen = torch.ones(batch_size, 1, device=self.device)
                    out_gen = self.model.dis_model(concat_data_dis_gen)
                    loss_gen = self.model.loss_fn(out_gen, y_gen)
                
                self.model.scaler.scale(loss_gen).backward()
                torch.nn.utils.clip_grad_norm_(self.model.gen_model.parameters(), max_norm=1)

                self.model.scaler.step(self.model.optimizer_gen)
                self.model.scaler.update()
                
                self.model.optimizer_gen.zero_grad()

                d_loss_total += loss_dis.item()
                g_loss_total += loss_gen.item()
                num_batches += 1
                bar.text(f"Epoch: {self.current_epoch}:{self.config.epochs} Gen/Dis Loss:{(g_loss_total/num_batches):.4f}/{(d_loss_total/num_batches):.4f}: ")
                bar()


        end_time = time.time()  # Record end time after training completes
        total_time = end_time - start_time  # Calculate total execution time

        return {'d_loss':d_loss_total/num_batches,'g_loss':g_loss_total/num_batches,'time': total_time}
