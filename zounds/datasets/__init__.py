"""
The datasets module provides access to some common sources of audio on the
internet.  In general, a dataset instance is an iterable of
:class:`zounds.soundfile.AudioMetaData` instances that can be passed to the
root node of an audio processing graph.
"""

from phatdrumloops import PhatDrumLoops
from internetarchive import InternetArchive
from freesound import FreeSoundSearch
from filesystem import Directory
from cache import DataSetCache
from ingest import ingest
