from celery.task import task,subtask
import os
import time
import numpy as np

LOCK_FN = 'lock.dat'

@task(name='queue.tasks.write')
def write(r):
    #while os.path.exists(LOCK_FN):
    #    time.sleep(1)
    
    #f = open(LOCK_FN,'w')
    #f.close()
    
    if not os.path.exists('results.txt'):
        f = open('results.txt','w')
    else:
        f = open('results.txt','a')
    f.write('%1.2f\n' % r)
    f.close()
    
    #os.remove(LOCK_FN)

@task(name='queue.tasks.mul')
def mul(a,b,callback=None):
    r = np.sum(a*b)
    if callback:
        subtask(callback).delay(r)

@task(name='queue.tasks.mw')
def mw(a,b):
    mul.delay(a,b,callback=subtask(write))