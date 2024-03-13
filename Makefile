# Define default make action
.DEFAULT_GOAL := help

# Project settings
PYTHON=python

# Commands
.PHONY: help setup train clean

help:
	@echo "Makefile for training a Conditional GAN on MNIST data"
	@echo ""
	@echo "Commands:"
	@echo "  setup    : Set up the python virtual environment and install dependencies."
	@echo "  train    : Start the training process with default or specified arguments."
	@echo "  clean    : Remove Python file artifacts and the virtual environment."


#optimizer= 'adam', 'rmsprop', 'sgd'
#	$(PYTHON) -m cganMNIST --epochs=200 --optimizer=adam --output-dir=adam-``
#	--gen_model_path=models/generator_epoch_100.pth --dis_model_path=models/discriminator_epoch_100.pth 

train-adam:
	output_dir=$$(date +'%y%m%d-%H%M%S'); \
	$(PYTHON) -m cganMNIST --workers=16 --batch=1000 --epochs=200 --dis-lr=0.0004 --gen-lr=.0002 --optimizer=adam --output-dir="outputs/adam-$$output_dir"

train-rmsprop:
#optimizer= 'adam', 'rmsprop', 'sgd'
	$(PYTHON) -m cganMNIST --epochs=200 --optimizer=rmsprop --gen_lr=.0004 --dis_lr=0.0001
#	--gen_model_path=models/generator_epoch_100.pth --dis_model_path=models/discriminator_epoch_100.pth 


train-sgd:
#optimizer= 'adam', 'rmsprop', 'sgd'
	$(PYTHON) -m cganMNIST --epochs=200 --optimizer=sgd --gen_lr=0.0001 --dis_lr=.0004
#	--gen_model_path=models/generator_epoch_100.pth --dis_model_path=models/discriminator_epoch_100.pth 

train-100:
	$(PYTHON) -m cganMNIST --epochs=800 --gen_model_path=models/generator_epoch_100.pth --dis_model_path=models/discriminator_epoch_100.pth 

clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_NAME)
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.png' -delete
	find . -type d -name '__pycache__' -exec rm -r {} +
	@echo "Clean up completed."

gif:
convert: unknown image property "%[frame]" @ warning/property.c/InterpretImageProperties/4088.
	convert -delay 10 -loop 0 $(for f in output/20240311-140546/images/img_*.png; do convert "$f" -bordercolor gray -border 10x10 -gravity SouthEast -annotate +10+10 %[frame] "$f"; done; echo output/20240311-140546/images/img_*.png) -delay 1000 output/20240311-140546/images/img_*.png assets/output.gif
