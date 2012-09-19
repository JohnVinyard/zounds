'''
The :code:`learn` module provides tools that make it easy to define a workflow 
for using unsupervised machine learning algorithms (supervised algorithms aren't 
currently supported, but may be in the future) to learn representations of your
data.  Once trained, your workflow is saved, and can be reused as a feature
extractor, by handing the workflow off to a :py:class:`zounds.analyze.feature.learned.Learned`
feature.  The workflow always includes these steps:
    
    * **Fetch** - Collect samples of the feature of interest from the datastore, \
    usually in a random fashion
    * **Preprocess** - Transform the data in a manner that makes the learning \
    algorithm (the next step) likely to work well.  In many cases, this step \
    is dependent on statistics of the data gathered during the **Fetch** phase, \
    and must be reproducible.  After training, this data is stored so that future \
    computations will always be consistent
    * **Learn** - Hand the data off to an unsupervised learning algorithm, such \
    as `k-means clustering <http://en.wikipedia.org/wiki/K-means_clustering>`_, or \
    an `autoencoder <http://en.wikipedia.org/wiki/Autoencoder>`_.

As an example, let's say we're interested in a feature that conveys some meaningful
information about spectral shape.  Perhaps we'd like to perform 
`k-means clustering <http://en.wikipedia.org/wiki/K-means_clustering>`_ on 
individual frames of :py:class:`zounds.analyze.feature.spectral.BarkBands`.

Assume we've defined a :py:class:`zounds.model.frame.Frames`-derived class which
details the features we're interested in...::
    
    class FrameModel(Frames):
        fft = Feature(FFT)
        bark = Feature(BarkBands, needs = fft, nbands = 100)

...and that we've analyzed a significant amount of audio.  We'd begin by 
instantiating a :py:class:`zounds.learn.pipeline.Pipeline` class::
    
    p = Pipeline(
        # this is a key than can be used to access our Pipeline once it's trained.
        # It can be anything you'd like, but it usually helps to give it a 
        # descriptive title
        'pipelines/bark/kmeans_500',
        # Each example we pull from the data store will consist of one frame
        # of the "bark" feature
        PrecomputedFeature(1,FrameModel.bark),
        # We're interested in spectral shape, and not overall loudness, so we'd
        # like to give each example unit-norm.  This means that the k-means 
        # clustering will differentiate between examples based on their shape,
        # and not wildly varying loudness
        UnitNorm(),
        # Finally, we'd like the k-means algorithm to learn 500 prototypical
        # examples, or clusters
        KMeans(500))

... and then training it. Don't worry about the second argument for now.  The first
argument means that we'll draw 10,000 pre-computed examples of bark bands from the
data store, at random::
    
    p.train(10000,lambda : True)

Once training is complete, we can always access our 
:py:class:`~zounds.learn.pipeline.Pipeline` in the following way, and apply it to
some new data::

    >>> p = Pipeline['pipelines/bark/kmeans_500'] # retrieve the Pipeline using our key
    >>> fake_barks = np.ones((25,100)) # mimic 25 frames of bark bands
    >>> kmeans = p(fake_barks) # preprocess the data, and assign each example to a cluster
    >>> kmeans.shape # each row of kmeans will have a single "1" in the position of the cluster assignment
    (25,500)

Now, we can incorporate the learned feature into our 
:py:class:`~zounds.model.frame.Frames`-derived class using a 
:py:class:`~zounds.analyze.feature.learned.Learned` extractor::
    
    class FrameModel(Frames):
        fft = Feature(FFT)
        bark = Feature(BarkBands, needs = fft, nbands = 100)
        kmeans = Feature(Learned, pipeline_id = 'pipelines/bark/kmeans_500',
                         dim = 500, dtype = np.uint8, needs = bark)
'''

