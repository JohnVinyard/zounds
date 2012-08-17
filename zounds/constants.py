from scikits.audiolab import available_file_formats as aff


available_file_formats = aff()
# Both 'aiff' and 'aif' file extensions are commonly used
available_file_formats.append('aif')


audio_key = 'audio'
id_key = '_id'
source_key = 'source'
external_id_key = 'external_id'

