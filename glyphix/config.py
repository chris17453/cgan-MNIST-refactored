import os
import yaml
from datetime import datetime

class ConfigManager:
    def __init__(self, parameters):
        if parameters==None:
            return

        # Defaults
        self.base_dir="output"
        self.created=None
        self.updated=None
        self.config_file = None
        self.output_dir = None
        self.optimizer  = 'adam'
        self.save_checkpoints = None
        self.checkpoint_interval = None
        self.gen_lr = .002
        self.dis_lr = .002
        self.epochs = None
        self.model_path = None
        self.log_file = None
        self.workers = 4
        self.batch_size = 100
        self.model_type = "MNIST"
        self.training_images= None
        self.train = None
        self.neurons = 256

        # load the params into the config variable...
        for key, value in parameters.items():
            setattr(self, key, value)
        self.print_info()
 
    def print_info(self):
        print("Current Configuration:")
        for key, value in vars(self).items():
            if key != 'config_file':  # Skip printing the config_file attribute
                print(f"  {key}: {value}")


    def build_output_directory(self):
        self.created = datetime.now().isoformat()
        if self.output_dir is None:
            self.output_dir= os.path.join(self.base_dir,datetime.now().strftime("%Y%m%d-%H%M%S"))

        self.image_dir=os.path.join(self.output_dir,"images")
        self.model_dir=os.path.join(self.output_dir,"models")
        self.log_file=os.path.join(self.output_dir,"log.txt")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        
    def load(self):
        try:
            with open(self.config_file, 'r') as file:
                loaded_data = yaml.safe_load(file)
                # Update the object's attributes with loaded data
                for key, value in loaded_data.items():
                    setattr(self, key, value)
        except FileNotFoundError:
            print(f"No configuration file found at '{self.config_file}'. Starting with default settings.")

    def update_attributes(self):
        self.output_dir = self.get('output_dir')
        self.optimizer = self.get('optimizer')
        self.save_checkpoints = self.get('save_checkpoints')
        self.checkpoint_interval = self.get('checkpoint_interval')
        self.gen_lr = self.get('gen_lr')
        self.dis_lr = self.get('dis_lr')
        self.epochs = self.get('epochs')
        self.model_path = self.get('model_path')
        self.model_type = self.get('model_type')
        try:
            self.neurons    = self.get('neurons')
        except:
            pass

    def save(self):
        # Update the 'updated' attribute with the current time
        self.updated = datetime.now().isoformat()

        # Prepare the data to save, excluding specific attributes
        data_to_save = {k: v for k, v in self.__dict__.items() if k != 'config_file'}

        # Open the file and save the YAML
        with open(self.config_file, 'w') as file:
            yaml.safe_dump(data_to_save, file, default_flow_style=False)

    def load_or_save(self):
       # Override any parameters with the config file.
        if self.config_file and os.path.exists(self.config_file)==True:
            self.load()
        else:
            self.build_output_directory()
            self.config_file=os.path.join(self.output_dir,"config.yaml")
            self.save()