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

def main():
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Train a Conditional GAN on MNIST or EMNST data.')
    parser.add_argument('--model', type=str, default='MNIST', choices=['MNIST', 'EMNIST'], help='Dataset to use')
    parser.add_argument('--output-dir', type=str,  help='Directory to save all output_data')
    parser.add_argument('--workers', type=int, default=4,  help='The number of threads for dataloaders')
    parser.add_argument('--batch', type=int, default=100,  help='The batch size for the dataloader')
    parser.add_argument('--save-checkpoints', action='store_true', help='Flag to save model checkpoints.')
    parser.add_argument('--checkpoint-interval', type=int, default=100, help='Interval (in epochs) between saving model checkpoints.')
    parser.add_argument('--gen_model-path', type=str, default=None, help='Path to the pre-trained generator model to load.')
    parser.add_argument('--dis_model-path', type=str, default=None, help='Path to the pre-trained discriminator model to load.')
    parser.add_argument('--epochs', type=int, default=800, help='Number of epochs to train the model.')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'rmsprop', 'sgd'], help='Optimizer for training: adam, rmsprop, or sgd.')
    parser.add_argument('--gen-lr', type=float, default=0.0002, help='Learning rate for the generator.')
    parser.add_argument('--dis-lr', type=float, default=0.0002, help='Learning rate for the discriminator.')
    parser.add_argument('--config', type=str, required=False, default=None, help='Path to a YAML configuration file.')

    args = parser.parse_args()


    # Disable output buffering for sys.stdout
    generate_welcome_ascii_art("glyphix",font="ansi_shadow")
    # Parse the arguments
    trainer = glyphix(
        output_dir          = args.output_dir,
        save_checkpoints    = args.save_checkpoints,
        checkpoint_interval = args.checkpoint_interval,
        gen_model_path      = args.gen_model_path,
        dis_model_path      = args.dis_model_path,
        epochs              = args.epochs,
        optimizer           = args.optimizer,
        gen_lr              = args.gen_lr,  
        dis_lr              = args.dis_lr,
        config              = args.config,
        workers             = args.workers,
        batch_size          = args.batch,
        model_type          = args.model
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
            


 
