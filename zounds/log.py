import logging
import logging.handlers

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='/temp/myapp.log',
                    filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
root_logger = logging.getLogger('zounds')
root_logger.addHandler(console)

rfile = logging.handlers.RotatingFileHandler(\
                            'log/log.txt',maxBytes=1*1000*1000,backupCount = 5)
rfile.setLevel(logging.DEBUG)
root_logger.addHandler(rfile)