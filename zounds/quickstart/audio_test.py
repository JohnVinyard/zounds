from zounds.pattern import *
import numpy as np
from time import sleep

latency = 0.25 * 1e6
interval = 1. * 1e6

def schedule(buf,ss,sts,times):

    now = usecs()
    times = times.astype(np.float32)
    times.sort()
    times = (times * 1e6) + latency + now


    cutoff = now + interval + latency
    index = times.searchsorted(cutoff)
    
    for i in xrange(index):
        put(buf[i],ss,sts,times[i])
    times = times[index:]
    buf = buf[index:]
    
    while True:
        sleep(interval / 1e6)
        now = usecs()
        cutoff = now + interval + latency
        index = times.searchsorted(cutoff)
        for i in xrange(index):
            put(buf[i],ss,sts,times[i])
        times = times[index:]
        buf = buf[index:]
        print times.size
        if not times.size:
            break
    

if __name__ == '__main__':
    a = np.random.random_sample(256)
    a -= .5
    a *= .1
    a = a.astype(np.float32)

    b = a.copy() * .1
    
    start()
    sleep(1)

    times = []
    bufs = []
    for i,t in enumerate(np.arange(0,100,.14)):
        buf = b if i % 4 else a
        times.append(t)
        bufs.append(buf)
    times = np.array(times,dtype = np.float32)


    schedule(bufs,0,256,times)
    