import logging.config
import logging

logging.basicConfig(filename='../logs/logging.conf', level=logging.DEBUG)
logging.config.fileConfig('../logs/logging.conf',
        defaults={'logfilename': '../logs/neural.log'},
        disable_existing_loggers=False)

