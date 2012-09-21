'''
The :py:module:`zounds.data` module concerns itself with persisting entities
defined in the :py:module:`zounds.model` module.

Storage options for :py:class:`~zounds.model.pipeline.Pipeline` and 
:py:class:`~zounds.model.framesearch.Framesearch` instances are limited to a
simple 
`cPickle <http://docs.python.org/release/2.6/library/pickle.html#module-cPickle>`_ + filesystem strategy.
 
There are currently two options for the persistence of 
:py:class:`~zounds.model.frame.Frames`-derived instances: the 
:py:class:`~zounds.data.frame.pytables.PyTablesFrameController`, and the 
:py:class:`~zounds.data.frame.pytables.FileSystemFrameController`.

The :py:class:`~zounds.data.frame.pytables.PyTablesFrameController` persists 
data in the `hdf5 <http://en.wikipedia.org/wiki/Hierarchical_Data_Format>`_ format 
and uses the excellent `PyTables <http://www.pytables.org/moin>`_ library to 
access it.  The :py:class:`~zounds.data.frame.pytables.PyTablesFrameController` 
tends to be a bit faster than the 
:py:class:`~zounds.data.frame.pytables.FileSystemFrameController` for data 
access, but has the major drawback that concurrent writes are a no-no.  This 
means that the embarrassingly parallel task of analyzing many files at once
and persisting the computed features must be limited to a single process.

The :py:class:`~zounds.data.frame.pytables.FileSystemFrameController`, as 
mentioned previously, is a bit slower for data access, but allows multi-process
analysis and database updates .  This can significantly speed up the analysis 
of large numbers of audio files.

Both classes implement the :py:class:`~zounds.data.frame.frame.FrameController` 
API.
'''