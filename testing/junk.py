
  
  
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Train a Conditional GAN on MNIST or EMNST data.')
    parser.add_argument('--model', type=str, default='MNIST', choices=['MNIST', 'EMNIST'], help='Dataset to use')
    parser.add_argument('--output-dir', type=str,  help='Directory to save all output_data')
    parser.add_argument('--workers', type=int, default=4,  help='The number of threads for dataloaders')
    parser.add_argument('--batch', type=int, default=100,  help='The batch size for the dataloader')
    parser.add_argument('--save-checkpoints', action='store_true', help='Flag to save model checkpoints.')
    parser.add_argument('--checkpoint-interval', type=int, default=100, help='Interval (in epochs) between saving model checkpoints.')
    parser.add_argument('--model-path', type=str, default=None, help='Path to the pre-trained generator model to load.')
    parser.add_argument('--epochs', type=int, default=800, help='Number of epochs to train the model.')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'rmsprop', 'sgd'], help='Optimizer for training: adam, rmsprop, or sgd.')
    parser.add_argument('--gen-lr', type=float, default=0.0002, help='Learning rate for the generator.')
    parser.add_argument('--dis-lr', type=float, default=0.0002, help='Learning rate for the discriminator.')
    parser.add_argument('--config', type=str, required=False, default=None, help='Path to a YAML configuration file.')
    parser.add_argument('--training-images', type=int, required=False, default=None, help='Interval to save training images.')
    
    args = parser.parse_args()
