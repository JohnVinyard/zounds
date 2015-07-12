from flow import Node,Aggregator
import os
from soundfile import SoundFile
import requests
import json

class AudioMetaData(object):
    
    def __init__(\
         self, 
         uri = None, 
         samplerate = None, 
         channels = None, 
         licensing = None,
         description = None,
         tags = None):
        
        super(AudioMetaData,self).__init__()
        self.uri = uri
        self.samplerate = samplerate
        self.channels = channels
        self.licensing = licensing
        self.description = description
        self.tags = tags

class AudioMetaDataEncoder(Aggregator,Node):
    
    content_type = 'application/json'
    
    def __init__(self, needs = None):
        super(AudioMetaDataEncoder,self).__init__(needs = needs)
    
    def _process(self,data):
        yield json.dumps({
               'uri' : data.uri.url \
                    if isinstance(data.uri,requests.Request) else data.uri,
               'samplerate' : data.samplerate,
               'channels' : data.channels,
               'licensing' : data.licensing,
               'description' : data.description,
               'tags' : data.tags
           })

class MetaData(Node):
    
    def __init__(\
         self, 
         needs = None, 
         freesound_api_key = None):
        
        super(MetaData,self).__init__(needs = needs)
        self._freesound_api_key = freesound_api_key
    
    def _process(self, data):
        uri = data
        if os.path.exists(uri):
            sf = SoundFile(uri)
            yield AudioMetaData(\
                 uri = uri, 
                 samplerate = sf.samplerate, 
                 channels = sf.channels)
        elif 'freesound.org' in uri:
            params = {'api_key' : self._freesound_api_key}
            meta = requests.get(\
                uri, params = params).json()
            req = requests.Request(\
               'GET', meta['serve'], params = params)
            yield AudioMetaData(\
                 uri = req,
                 samplerate = meta['samplerate'],
                 channels = meta['channels'],
                 licensing = meta['license'],
                 description = meta['description'],
                 tags = meta['tags'])
        else:
            raise Exception('Cannot handle uri {uri}'.format(**locals()))
            