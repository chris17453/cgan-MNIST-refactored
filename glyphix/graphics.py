import os
import torch
import numpy as np
from PIL import Image
from .common import create_base_dir_of_file

from .common import loop_filename

class graphics:
    def __init__(self,model,width,height,filename):
        self.model=model
        self.width=width
        self.height=height
        self.filename=filename
        self.new_image()

    def new_image(self):
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



    class text:
        def __init__(self,x,y,width=0,height=0,text=None):
            self.x=x
            self.y=y
            self.width=width
            self.height=height
            self.text=text
            self.y2=y
            self.x2=x
            self.x_pos=x
            self.y_pos=y
        
        def append(self,char,width):
              if self.text==None:
                  self.text=[]
              self.text.append(char)
              self.add_width(width)
              self.x_pos+=width

        def add_height(self,height):
            self.height+=height
            self.y2+=height

        def add_width(self,width):
            self.width+=width
            self.x2+=width


        def set_x(self,x):
            self.x=x
            self.x2=x+self.width
            self.x_pos=x

        def set_y(self,y):
            self.y=y
            self.y2=y+self.height
            self.y_pos=y            

        def __str__(self) :
            output= \
            f"x:       {self.x}     \n" +\
            f"y:       {self.y}     \n" +\
            f"y2:      {self.y2}     \n" +\
            f"x2:      {self.x2}     \n" +\
            f"width:   {self.width}     \n" +\
            f"height:  {self.height}     \n" +\
            f"x_pos:   {self.x_pos}     \n" +\
            f"y_pos:   {self.y_pos}     \n" +\
            f"text:    {self.text}     \n" 
            return output

    def generate_fragments(self,text,x_offset, y_offset):
        fragments=[]
        x_pos=x_offset
        y_pos=y_offset
        
        f=None
        for char in text:
            ## if we see a space.. lets just break this bad boy
            if char==' ':
                if f:
                    fragments.append(f)
                    f=None
                hidden=True
                x_pos+=self.model.char_width-2
            # if we see a new line break it as well
            elif char=='\n':
                if f:
                    fragments.append(f)
                    f=None
                x_pos+=x_offset
                y_pos+=self.model.char_height
                hidden=True
            
            # if we see a line return.. ignore it.. its a usless artifact of days gone
            elif char=='\n':
                hidden=True
            else:
                hidden=None


            # if we are off the page.. drop down a line
            if x_pos>self.width:
                x_pos=x_offset
                y_pos+=self.model.char_height
            
            # if we are off the page.. exit loop
            if y_pos > self.height:
                break  # Stop if we run out of room
            
            # if we cant see this character.. lets not add it to the array
            if hidden==True:
                continue

            if f==None:
                f=self.text(x=x_pos,y=y_pos,height=self.model.char_height)
            

            f.append(char,self.model.char_width)
            x_pos += self.model.char_width


            ##if the text does not begin at the start of the bounding area
            if f.x!=x_offset:
                # and the next character is wider than the bounding area... 
                if f.x_pos + self.model.char_width > self.width:
                    # Just move it down 1 line to the left
                    f.set_x(x_offset)
                    x_pos=f.x2
                    y_pos += self.model.char_height
                    f.set_y(y_pos)
                    if y_pos > self.height:
                        break  # Stop if we run out of room
            
            
            # if the text does start at the bounding area  (lets not elst this statment with the above or we will miss data changes in the loop)
            if f.x==x_offset:
                # and the next character is wider than the bounding area... 
                if f.x_pos + self.model.char_width > self.width:
                    #break the text and start a new one on the next line
                    fragments.append(f)

                    x_pos=x_offset
                    y_pos += self.model.char_height
                    f=None
                    if y_pos + self.model.char_height > self.height:
                        break  # Stop if we run out of room
            

        # if there is a straggler.. lets slap that bad boy back in the array
        if f:
            fragments.append(f)
        
        # return the bounty of parsed spacial bounding text fragments of justice
        return fragments



    def write_text(self,text,x_offset=0, y_offset=0):
        create_base_dir_of_file(self.filename)

        fragments=self.generate_fragments(text,x_offset, y_offset)


        for fragment in fragments:
            x_offset=fragment.x
            y_offset=fragment.y
            for char in fragment.text:
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
                else:
                    x_offset += self.model.char_width 

    def write_text_loop(self,text,x_offset=0, y_offset=0):
        for f in loop_filename(self.filename,20):
            self.new_image()
            self.write_text(text,x_offset,y_offset)
            self.save(f)
    

    def save(self,filename=None):
        if filename:
            self.image.save(filename)    
        else:
            self.image.save(self.filename)

