import os
import torch
from torchvision import transforms as T
from torchvision.datasets import EMNIST,MNIST
from tqdm import tqdm
from .data_transforms import CLAHETransform,HistogramEqualizationTransform

from torch.utils.data import Dataset

class PrecomputedDataset(Dataset):
    def __init__(self, filepath):
        # Load the preprocessed data and metadata
        loaded_data = torch.load(filepath)
        self.data = loaded_data['data']
        self.classes = loaded_data['classes']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return the preprocessed image and label
        return self.data[idx]
    
class PrecomputeDataset:
    def __init__(self, root_dir="data", model_type="MNIST"):
        self.root_dir = root_dir
        self.model_type = model_type
        self.transform = self.get_transform()
        self.dataset_class = MNIST if model_type == "MNIST" else EMNIST

    def get_transform(self):
        # Define the transformation pipeline
        if self.model_type == "MNIST":
            return T.Compose([
#                CLAHETransform(),
                #HistogramEqualizationTransform(),
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5)
            ])
        else:  # EMNIST
            return T.Compose([
                lambda img: T.functional.rotate(img, -90),
                lambda img: T.functional.hflip(img),
                #HistogramEqualizationTransform(),
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5)
            ])

    def preprocess_and_save(self):
        # Load the dataset without applying any transformations
        if self.model_type=="MINST":
            train_dataset = self.dataset_class(self.root_dir, train=True, download=True, transform=None)
            valid_dataset = self.dataset_class(self.root_dir, train=False, download=True, transform=None)
        else:
            train_dataset = self.dataset_class(self.root_dir,split="byclass", train=True, download=True, transform=None)
            valid_dataset = self.dataset_class(self.root_dir,split="byclass", train=False, download=True, transform=None)


        # Process and save training data
        self.process_and_save_subset(train_dataset, 'train')
        # Process and save validation data
        self.process_and_save_subset(valid_dataset, 'valid')


    def process_and_save_subset(self, dataset, subset_name):
        processed_data = []
        for img, label in tqdm(dataset, desc=f"Processing {subset_name} data"):
            # Apply the transformation
            img = self.transform(img)
            processed_data.append((img, label))

        dataset_info = {
            "data": processed_data,
            "classes": dataset.classes
        }

        # Save the processed data along with class information
              # Save the processed data
        torch.save(dataset_info, os.path.join(self.root_dir, f"{self.model_type.lower()}_{subset_name}_preprocessed.pt"))
        print(f"Saved {subset_name} data.")


    def load_precomputed_dataset(self, train=True):
        subset_name = 'train' if train else 'valid'
        filepath = os.path.join(self.root_dir, f"{self.model_type.lower()}_{subset_name}_preprocessed.pt")
        if os.path.exists(filepath):
            print ("Loading Pre computed Dataset")
            return PrecomputedDataset(filepath)
        else:
            print ("Building Pre computed Dataset")
            self.preprocess_and_save()
            if os.path.exists(filepath):
                return PrecomputedDataset(filepath)
            raise FileNotFoundError(f"{filepath} does not exist. Preprocessiong")    
        
    def load(self):
        train_dataset=self.load_precomputed_dataset(train=True)
        valid_dataset=self.load_precomputed_dataset(train=None)
        return train_dataset,valid_dataset
        

