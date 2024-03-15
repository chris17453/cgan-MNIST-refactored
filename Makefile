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
#	$(PYTHON) -m glyphix --epochs=200 --optimizer=adam --output-dir=adam-``
#	--gen_model_path=models/generator_epoch_100.pth --dis_model_path=models/discriminator_epoch_100.pth 

minst-text:
	$(PYTHON) -m glyphix \
		create-image \
		--text="999 8675309" \
		--width=800 \
		--height=100 \
		--output=assets/mnist-text.png \
		--model-path=outputs/emnist/240314-204122/models/EMNIST_epoch_300.pth
		
#		$outputs/mnist/240314-204342/models/MNIST_epoch_542.pth

eminst-text:
	$(PYTHON) -m glyphix \
		create-image \
		--text="1324567890 Life is just. Work that is well done is its own reward." \
		--width=800 \
		--height=100 \
		--output=assets/emnist-text.png \
		--model-path=outputs/emnist/240314-210720/models/EMNIST_epoch_799.pth

minst-adam:
	output_dir=$$(date +'%y%m%d-%H%M%S'); \
	$(PYTHON) -m glyphix train --checkpoint=100 --training-image=1 --model=MNIST --workers=4 --batch=1000 --epochs=800 --dis-lr=.0001 --gen-lr=.0004 --optimizer=adam --output-dir="outputs/mnist/$$output_dir"


eminst-adam:
	output_dir=$$(date +'%y%m%d-%H%M%S'); \
	$(PYTHON) -m glyphix train --checkpoint=100 --training-image=1 --model=EMNIST --workers=20 --batch=1000 --epochs=800 --dis-lr=.0001 --gen-lr=.0004 --optimizer=adam --checkpoint=100 --output-dir="outputs/emnist/$$output_dir"

train-rmsprop:
#optimizer= 'adam', 'rmsprop', 'sgd'
	$(PYTHON) -m glyphix --epochs=200 --optimizer=rmsprop --gen_lr=.0004 --dis_lr=0.0001
#	--gen_model_path=models/generator_epoch_100.pth --dis_model_path=models/discriminator_epoch_100.pth 


train-sgd:
#optimizer= 'adam', 'rmsprop', 'sgd'
	$(PYTHON) -m glyphix --epochs=200 --optimizer=sgd --gen_lr=0.0001 --dis_lr=.0004
#	--gen_model_path=models/generator_epoch_100.pth --dis_model_path=models/discriminator_epoch_100.pth 

train-100:
	$(PYTHON) -m glyphix --epochs=800 --gen_model_path=models/generator_epoch_100.pth --dis_model_path=models/discriminator_epoch_100.pth 

clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_NAME)
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.png' -delete
	find . -type d -name '__pycache__' -exec rm -r {} +
	@echo "Clean up completed."

