import os
import random

def generate_readable_filename():
    fruits = ["banana", "apple", "orange", "pear"]
    furniture = ["desk", "couch", "chair", "table"]
    electronics = ["lamp", "computer", "phone", "tv"]
    stationery = ["book", "pen", "notebook", "eraser"]
    
    choices = [random.choice(fruits), random.choice(furniture), random.choice(electronics), random.choice(stationery)]
    return "-".join(choices)

def create_base_dir_of_file(filename):
    base_dir = os.path.dirname(filename)
    if base_dir and base_dir!='':
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)


def print_info(obj):
    print("Current Configuration:")
    for key, value in vars(obj).items():
        if key != 'config_file':  # Skip printing the config_file attribute
            print(f"  {key}: {value}")

