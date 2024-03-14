import os
import math
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.datasets import EMNIST,MNIST
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image
import numpy as np
from PIL import ImageFilter, Image
import cv2


from .generator import Generator,Discriminator
from .dataset import CustomTransform



    
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
        self.create_char_mapping()

    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
 
    def configure(self):
 
        custom_transform = CustomTransform(model_type=self.config.model_type)
        train_dataset, valid_dataset = custom_transform.load_datasets()

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
        self.num_classes_data= len(train_dataset.classes)

        self.fc_neuron = 256

        self.logger.log(f'Height: {self.char_height}')
        self.logger.log(f'Width: {self.char_width}')
        self.logger.log(f'Neuron: {self.fc_neuron}')

        # Initialize models
        #self.gen_model = Generator(self.char_height,self.char_width,self.fc_neuron).to(self.device)
        #self.dis_model = Discriminator(self.char_height,self.char_width,self.fc_neuron).to(self.device)
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
        self.save_fake_data()

    def configure_optimizers(self):
        if self.config.optimizer == 'adam':
            self.optimizer_gen = torch.optim.Adam(self.gen_model.parameters(), lr=self.config.gen_lr,  betas=(0.5, 0.999))
            self.optimizer_dis = torch.optim.Adam(self.dis_model.parameters(), lr=self.config.dis_lr,  betas=(0.5, 0.999))
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

  
    def autosize_grid(self,total_count):
        """
        Automatically determine the size of a square-ish grid needed to fit all items.
        
        Parameters:
        - total_count: int, the total number of items to fit in the grid.
        
        Returns:
        - rows: int, the number of rows in the grid.
        - cols: int, the number of columns in the grid.
        """
        cols = math.ceil(math.sqrt(total_count))
        rows = math.ceil(total_count / cols)
        
        return rows, cols
    

    def save_training_image(self, epoch):
        # Ensure the save directory exists
        os.makedirs(self.config.image_dir, exist_ok=True)

        # creating images of numbers one to ten
        target_label = torch.tensor(range(self.num_classes_data))
        image_tensors = []  # Use a list to collect image tensors

        for number in target_label:
            with torch.no_grad():
                bs = 1
                z_test = torch.randn(bs, self.z_dim).to(self.device)
                target_onehot = torch.nn.functional.one_hot(number, self.num_classes_data).float().to(self.device) # Ensure it's float for compatibility
                target_onehot = torch.unsqueeze(target_onehot, dim=0)
                concat_target_data = torch.cat((target_onehot, z_test), dim=1)
                output = self.gen_model(concat_target_data)
                image_tensors.append(output.view(bs, self.image_channel, self.char_height, self.char_width))

        outputs = torch.cat(image_tensors, dim=0)  # Concatenate all collected tensors
        outputs = outputs * 0.5 + 0.5  # Assuming normalization was mean=0.5, std=0.5

        grid=self.autosize_grid(self.num_classes_data)
        # display the created images
        img_grid = make_grid(outputs, nrow=grid[0])
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

   
    def save_fake_data(self):
        x, _ = next(iter(self.data_loader))
        grid = self.autosize_grid(self.config.batch_size)

        # Reverse normalization here: Assuming the normalization was mean=0.5, std=0.5
        # Adjust for images normalized differently
        x = x * 0.5 + 0.5

        img_grid = make_grid(x, nrow=grid[0])
        
        # Convert the tensor to [0, 1] range if not already
        img = img_grid.cpu().detach().numpy()
        img = np.transpose(img, (1, 2, 0))
        
        # Convert to [0, 255] for saving as an image file
        img = (img * 255).astype(np.uint8)
        
        # Convert numpy array to PIL Image
        img = Image.fromarray(img)

        filename = os.path.join(self.config.image_dir, "sample_data.png")
        img.save(filename)

    def create_char_mapping(self):
        # Create sequences for digits, uppercase, and lowercase letters
        digits = [str(i) for i in range(10)]
        uppercase_letters = [chr(i) for i in range(ord('A'), ord('Z')+1)]
        lowercase_letters = [chr(i) for i in range(ord('a'), ord('z')+1)]

        # Combine all sequences and map each to an index
        all_chars = digits + uppercase_letters + lowercase_letters
        self.char_mapping = {char: index for index, char in enumerate(all_chars)}

    def write_file(self,text,x_offset, y_offset,filename):
        os.makedirs(self.config.image_dir, exist_ok=True)
        
        # Background image dimensions
        bg_width, bg_height = 1024, 512  # Example dimensions, adjust as needed
        background_image = Image.new('RGB', (bg_width, bg_height), color = (0, 0, 0))

        for char in text:
            if char in self.char_mapping:
                target_label = torch.tensor([self.char_mapping[char]])
                with torch.no_grad():
                    bs = 1
                    z_test = torch.randn(bs, self.z_dim).to(self.device)
                    target_onehot = torch.nn.functional.one_hot(target_label, self.num_classes_data).float().to(self.device)
                    target_onehot = torch.unsqueeze(target_onehot, dim=0)
                    concat_target_data = torch.cat((target_onehot, z_test), dim=1)
                    output = self.gen_model(concat_target_data)
                    char_image_tensor = output.view(bs, self.image_channel, self.char_height, self.char_width)

                    # Convert tensor to PIL Image
                    char_img = char_image_tensor.cpu().detach().numpy()
                    char_img = np.transpose(char_img, (0, 2, 3, 1)).reshape(self.char_height, self.char_width, self.image_channel)
                    char_img = (char_img * 255).astype(np.uint8)
                    char_img_pil = Image.fromarray(char_img)

                    # Paste the character image onto the background
                    background_image.paste(char_img_pil, (x_offset, y_offset))

                    # Update x_offset for the next character
                    x_offset += self.char_width
                    if x_offset + self.char_width > bg_width:
                        x_offset = 0
                        y_offset += self.char_height
                        if y_offset + self.char_height > bg_height:
                            break  # Stop if we run out of room

        # Save the composed image
        background_image.save(filename)

