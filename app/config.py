from analyze.feature import FFT,Loudness


# Audio Config
samplerate = 44100.
windowsize = 2048.
stepsize = windowsize / 2.


# Data backend
frame_controller = None
pattern_controller = None
learn_controller = None


# RowModel definition
fft = FFT()
loudness = Loudness(needs=fft)
rbm = Learned(Pipeline['rbm'],needs=fft)

class RowModel(object):
    fft = Feature(fft,store=False)
    loudness = Feature(loundess,store=True)
    rbm = Feature(rbm,store=True)
    
    




    
    
               
               

