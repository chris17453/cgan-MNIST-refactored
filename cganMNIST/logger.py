import sys
from contextlib import nullcontext, redirect_stdout

class Logger:
    def __init__(self, filepath, stdout=False):
        """
        Initializes the logger.

        :param filepath: Path to the log file.
        :param stdout: If True, log messages will also be printed to stdout.
        """
        self.filepath = filepath
        self.stdout = stdout


    def log(self, message,no_return=None,dont_log=None,no_move_cursor=None,stdout=True):
        """Appends a log message to the file and optionally prints to stdout."""
        if dont_log==None:
            with open(self.filepath, 'a') as file:
                file.write(message + '\n')
        if self.stdout and stdout:
            with redirect_stdout(sys.stdout):
                if no_move_cursor:
                    # Save the cursor position
                    sys.stdout.write("\033[s")

                if no_return:
                    sys.stdout.write(f"{message}")
                else:
                    sys.stdout.write(f"{message}\n\r")

                if no_move_cursor:
                    # Restore the cursor position
                    sys.stdout.write("\033[u")

                sys.stdout.flush()  
  
