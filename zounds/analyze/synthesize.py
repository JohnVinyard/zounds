import numpy as np
from scikits.audiolab import play


class WindowedAudioSynthesizer(object):
    '''
    A very simple synthesizer that assumes windowed, but otherwise raw frames
    of audio.
    '''
    def __init__(self,windowsize,stepsize):
        object.__init__(self)
        self.windowsize = windowsize
        self.stepsize = stepsize
        
        # libvoribs 1.3.1 is causing a segmentation fault when writing samples
        # near this number (regardless of sample rate)
        self.vorbis_seg_fault_size = 45 * 44100
        # Writing samples to a ogg vorbis file in chunks eliminates the segfault,
        # and is faster in some cases (especially for longer files), but introduces
        # small, but noticeable clicks at the chunk boundaries.  Libvorbis 
        # seems to not be "crosslapping" at write boundaries. 
        self.vorbis_chunk_size = 25 * 44100
    
    def _vorbis_write(self,frames,sndfile,output):
        cs = self.vorbis_chunk_size
        waypoint = 0
        for i,f in enumerate(frames):
            start = i * self.stepsize
            stop = start + self.windowsize
            output[start : stop] += f
            if (stop - waypoint) > cs:
                # a sndfile was passed in, and we have enough data to write
                # a chunk of audio to the file
                sndfile.write_frames(output[waypoint : stop])
                # update the end of the last chunk written
                waypoint = stop
        
        if sndfile:
            # write any remaining data to disk
            sndfile.write_frames(output[waypoint:])
        
        return output
    
    def __call__(self,frames,sndfile = None):
        '''
        Parameters
            frames  - a two dimensional array containing frames of audio samples
            sndfile - Optional. A scikits.audiolab.Sndfile instance to which the samples
                      should be written.
        
        Returns
            output - a 1D array containing the rendered audio
        '''
        output = np.zeros(self.windowsize + (len(frames) * self.stepsize))
        
        if sndfile and 'ogg' == sndfile.file_format:
            # The caller has requested that output audio be written to an ogg
            # vorbis file.  libvorbis 1.3.1 is currently causing segmentation faults
            # for larger files, so we'll need to handle this specially.
            return self._vorbis_write(frames, sndfile, output)
    
        
        for i,f in enumerate(frames):
            start = i * self.stepsize
            stop = start + self.windowsize
            output[start : stop] += f

    
        if sndfile:
            sndfile.write_frames(output)
        
        return output
            
    def play(self,audio):
        output = self(audio)
        return self.playraw(output)
    
    def playraw(self,audio):
        try:
            play(np.tile(audio,(2,1)) * .2)
        except KeyboardInterrupt:
            pass

# TODO: FFT synthesizer
# TODO: DCT synthesizer
# TODO: ConstantQ synthesizer