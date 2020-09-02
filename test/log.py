import logging
import logging.handlers
import torch
import datetime

class logger():
    def __init__(self, split):
        self.split = split
        self.logger = logging.getLogger('mylogger')
        self.logger.setLevel(logging.DEBUG)

        f_handler = logging.FileHandler('info-'+self.split+'.log')
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))

        self.logger.addHandler(f_handler)

    def get_info(self, infor):
        self.logger.info(infor)

a = 'string'
print(a)
log = logger('tttt')
log.get_info("This log works!"+a)  # a必须是string类型的