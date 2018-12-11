# MIT License
#
# Copyright (c) 2018 Pedro Virgilio Basilio Jeronymo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
My Logger

Auxiliary class based in python's default logging class.
More info on github.com/pedrovbj/SVHN-ML
'''


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
