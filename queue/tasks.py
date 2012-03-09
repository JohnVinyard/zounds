from celery.task import task


@task(name='queue.tasks.mul')
def mul(a,b):
    return a*b