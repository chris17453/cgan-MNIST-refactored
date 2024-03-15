import os
import torch
import numpy as np
from PIL import Image
from .common import create_base_dir_of_file


class graphics:
    def __init__(self,model,width,height,filename):
        self.model=model
        self.width=width
        self.height=height
        self.filename=filename
        self.image = Image.new('RGB', (self.width, self.height), color = (0, 0, 0))


    def calculate_bounding_box(self,image_pil):
        """
        Calculate the bounding box of significant content in an image based on brightness.
        
        Parameters:
        - image_pil: PIL.Image, the input image.
        
        Returns:
        - A tuple representing the bounding box (left, upper, right, lower).
        """
        # Convert the image to a binary image based on a brightness threshold.
        # Adjust the threshold value based on your specific needs (e.g., 128 for a middle brightness in grayscale)
        threshold = 20
        binary_image = image_pil.point(lambda p: p > threshold and 255)
        
        # Convert to numpy array for analysis
        binary_array = np.array(binary_image)
        
        # Find rows and columns where there are pixels brighter than the threshold
        rows = np.any(binary_array, axis=1)
        cols = np.any(binary_array, axis=0)
        if not rows.any() or not cols.any():  # Image is completely dark or below threshold
            return (0, 0, image_pil.width, image_pil.height)  # Return the entire image as the bounding box
        
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        # Convert these to a bounding box tuple
        bounding_box = (xmin, ymin, xmax + 1, ymax + 1)
        
        return bounding_box
    
    def write_text(self,text,x_offset, y_offset):
        create_base_dir_of_file(self.filename)
        for char in text:
            if char in self.model.char_mapping:
                target_label = torch.tensor(self.model.char_mapping[char])
                with torch.no_grad():
                    bs = 1

                    z_test = torch.randn(bs, self.model.z_dim).to(self.model.device)
                    target_onehot = torch.nn.functional.one_hot(target_label, self.model.num_classes_data).float().to(self.model.device)
                    target_onehot = torch.unsqueeze(target_onehot, dim=0)
                    concat_target_data = torch.cat((target_onehot, z_test), dim=1)
                    output = self.model.gen_model(concat_target_data)
                    output = (output * 0.5 + 0.5)*255  # Assuming normalization was mean=0.5, std=0.5
                    char_image_tensor = output.view(bs, self.model.image_channel, self.model.char_height, self.model.char_width)

                    # Convert tensor to PIL Image
                    char_img = char_image_tensor.cpu().detach().numpy().astype(np.uint8)
                    #print( char_img.ndim )
                   
                    # Correct reshaping based on your dimensions; assuming grayscale image for simplicity
                    if char_img.ndim == 4:  # (batch_size, channels, height, width)
                        char_img = np.squeeze(char_img)  # This removes dimensions of size 1
                    char_img_pil = Image.fromarray(char_img,mode="L")
                    
                    bounding_area=self.calculate_bounding_box(char_img_pil)
                    
                    # Paste the character image onto the background
                    self.image.paste(char_img_pil, (x_offset-bounding_area[0], y_offset))

                    # Update x_offset for the next character
                    #x_offset += self.model.char_width

                    x_offset += bounding_area[2]-bounding_area[0]+5
                    if x_offset + self.model.char_width > self.width:
                        x_offset = 0
                        y_offset += self.model.char_height
                        if y_offset + self.model.char_height > self.height:
                            break  # Stop if we run out of room
            else:
                x_offset += self.model.char_width 


    def save(self):
        self.image.save(self.filename)
