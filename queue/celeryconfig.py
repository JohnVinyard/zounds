# List of modules to import when celery starts.
CELERY_IMPORTS = ("tasks", )

## Result store settings.
CELERY_RESULT_BACKEND = "database"
CELERY_RESULT_DBURI = "sqlite:///mydatabase.db"

## Broker settings.
BROKER_URL = "amqp://guest:guest@localhost:5672//"

## Worker settings
## If you're doing mostly I/O you can have more processes,
## but if mostly spending CPU, try to keep it close to the
## number of CPUs on your machine. If not set, the number of CPUs/cores
## available will be used.
#CELERYD_CONCURRENCY = 10

#CELERY_ANNOTATIONS = {"tasks.add": {"rate_limit": "10/s"}}