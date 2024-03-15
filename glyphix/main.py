import termios
import tty
import sys
import select
import argparse


from .glyphix import glyphix
from .messages import generate_welcome_ascii_art, display_goodbye_message



def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        [i, _, _] = select.select([sys.stdin], [], [], 0)
        if i:
            ch = sys.stdin.read(1)
        else:
            ch = ''
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def train(args):
    # Disable output buffering for sys.stdout
    generate_welcome_ascii_art("glyphix",font="ansi_shadow")
    # Parse the arguments
    trainer = glyphix(
        output_dir          = args.output_dir,
        checkpoint_interval = args.checkpoint,
        model_path          = args.model_path,
        epochs              = args.epochs,
        optimizer           = args.optimizer,
        gen_lr              = args.gen_lr,  
        dis_lr              = args.dis_lr,
        config              = args.config,
        workers             = args.workers,
        batch_size          = args.batch,
        model_type          = args.model,
        training_images     = args.training_images,
        train               = True
    )
    # print the info after configure, because we calculate epoch start index after model load if requested
    trainer.print_info()
    t_thread=trainer.start()
    print("Press Ctrl+S to save, Ctrl+X to exit, or any other key to continue.")
    while True:
         
        char = getch()
        if char == '\x13':  # Ctrl+S
            print("Save Requested")
            trainer.queue_save()
        elif char == '\x18' or char =='\x03' or trainer.running==False:  # Ctrl+X
            print("Exiting")
            trainer.stop()
            t_thread.join()
            display_goodbye_message()
            break
        elif char:
            print("Unknown Command...")
            pass

def create_image(args):
    from.model import model
    from .graphics import graphics
    from .config import ConfigManager
    config=ConfigManager({'text':args.text,
                        'x_offset':args.x,
                        'y_offset':args.y,
                        'width':args.width,
                        'height':args.height,
                        'model_path':args.model_path,
                        'filename':args.output})
    config.print_info()
    m=model(config)

    gr=graphics( model       = m,
                width       = args.width,
                height      = args.height,
                filename    = args.output )
    gr.write_text(text=args.text,x_offset=args.x,y_offset=args.y)
    gr.save()



def main():
    # Define the main argument parser
    parser = argparse.ArgumentParser(description='Tool for training Conditional GANs and creating images.')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Subparser for the "train" command
    train_parser = subparsers.add_parser('train', help='Train a Conditional GAN model')
    train_parser.add_argument('--model', type=str, default='MNIST', choices=['MNIST', 'EMNIST'], help='Dataset to use')
    train_parser.add_argument('--output-dir', type=str, required=True, help='Directory to save all output data')
    train_parser.add_argument('--workers', type=int, default=4, help='The number of threads for data loaders')
    train_parser.add_argument('--batch', type=int, default=100, help='The batch size for the data loader')
    train_parser.add_argument('--checkpoint', type=int, default=None, help='Interval (in epochs) between saving model checkpoints, default None.')
    train_parser.add_argument('--model-path', type=str, help='Path to the pre-trained generator model to load.')
    train_parser.add_argument('--epochs', type=int, default=800, help='Number of epochs to train the model.')
    train_parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'rmsprop', 'sgd'], help='Optimizer for training.')
    train_parser.add_argument('--gen-lr', type=float, default=0.0002, help='Learning rate for the generator.')
    train_parser.add_argument('--dis-lr', type=float, default=0.0002, help='Learning rate for the discriminator.')
    train_parser.add_argument('--config', type=str, help='Path to a YAML configuration file.')
    train_parser.add_argument('--training-images', type=int, help='Interval to save training images.')

    # Subparser for the "create-image" command
    create_image_parser = subparsers.add_parser('create-image', help='Create an image with specified text')
    create_image_parser.add_argument('--text', type=str, required=True, help='Text to overlay on the image')
    create_image_parser.add_argument('--width', type=int, default=800, help='Width of the image')
    create_image_parser.add_argument('--height', type=int, default=600, help='Height of the image')
    create_image_parser.add_argument('--x', type=int, default=0, help='X position of text')
    create_image_parser.add_argument('--y', type=int, default=0, help='Y position of text')
    
    create_image_parser.add_argument('--output', type=str, default="text.png", help='filename of the image to save')
    create_image_parser.add_argument('--model-path', type=str, default=None,required=True, help='file location of the model to use')
    

    args = parser.parse_args()

    # Here, you would add your logic based on the command
    if args.command == 'train':
        train(args)
    elif args.command == 'create-image':
        c=create_image(args)
        pass


                


 
