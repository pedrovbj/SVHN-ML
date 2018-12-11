import sys
import logging

class MyLogger():
    def __init__(self, path='results/log.txt', save=True):
        self.logger = logging.getLogger('My Logger')
        self.logger.setLevel(logging.DEBUG)

        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.flush = sys.stdout.flush
        consoleHandler.terminator = ''
        self.logger.addHandler(consoleHandler)

        if save:
            fileHandler = logging.FileHandler(path)
            fileHandler.terminator = ''
            self.logger.addHandler(fileHandler)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)
        self.logger.handlers[0].flush()

    def shutdown(self):
        logging.shutdown()
        self.logger.handlers.clear()

    def get_logger(self):
        return self.logger

if __name__ == '__main__':
    logger = MyLogger('/tmp/hello.txt')
    logger.debug('Hello,')
    logger.debug(' World!\r\n')
    logger.shutdown()
