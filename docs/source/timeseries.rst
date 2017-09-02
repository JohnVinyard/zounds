Timeseries
==========
.. automodule:: zounds.timeseries
.. currentmodule:: zounds.timeseries

Audio Samples
-------------
.. autoclass:: AudioSamples
    :members:

The Time Dimension
------------------
.. autoclass:: TimeDimension
    :members:
.. autoclass:: TimeSlice
    :members:

Sample Rates
------------
.. autoclass:: SR96000
.. autoclass:: SR48000
.. autoclass:: SR44100
.. autoclass:: SR22050
.. autoclass:: SR11025
.. autoclass:: SampleRate
    :members:

Durations
---------
Zounds includes several convenience classes that make it possible to create
time durations as :class:`numpy.timedelta64` instances without remembering or
using magic strings to designate units.

.. autoclass:: Hours
.. autoclass:: Minutes
.. autoclass:: Seconds
.. autoclass:: Milliseconds
.. autoclass:: Microseconds
.. autoclass:: Nanoseconds
.. autoclass:: Picoseconds


