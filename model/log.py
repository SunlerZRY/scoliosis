import logging
import logging.handlers


class logger():
    def __init__(self):
        self.logger = logging.getLogger('mylogger')
        self.logger.setLevel(logging.DEBUG)

        f_handler = logging.FileHandler('info-13.log')
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))

        self.logger.addHandler(f_handler)

    def get_info(self, infor):
        self.logger.info(infor)
