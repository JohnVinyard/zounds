import logging
import logging.handlers

print __name__

record_fmt = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
date_fmt = '%m-%d %H:%M'

formatter = logging.Formatter(record_fmt,datefmt = date_fmt)

# the parent logger for the entire zounds library
root_logger = logging.getLogger('zounds')

# one megabyte, in bytes
mb = 1e6
# the name of the most current log file
log_file_name = 'log/log.txt'
# the number of old log files to keep
backups = 5
# the rotating log file handler
rfile = logging.handlers.RotatingFileHandler(\
                            log_file_name,maxBytes=mb,backupCount = backups)
# the console handler
console = logging.StreamHandler()

# for now, log everything to both the console and the log file
rfile.setLevel(logging.DEBUG)
console.setLevel(logging.DEBUG)
rfile.setFormatter(formatter)
console.setFormatter(formatter)

root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(rfile)
root_logger.addHandler(console)