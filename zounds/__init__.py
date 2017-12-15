__version__ = '0.25.12'

from timeseries import \
    Hours, Minutes, Seconds, Milliseconds, Microseconds, Picoseconds, \
    SR11025, SR16000, SR22050, SR44100, SR48000, SR96000, HalfLapped, Stride, \
    TimeSlice, VariableRateTimeSeries, VariableRateTimeSeriesFeature, \
    SampleRate, AudioSamples, TimeDimension, audio_sample_rate, \
    ConstantRateTimeSeries, nearest_audio_sample_rate

from soundfile import \
    MetaData, AudioMetaData, AudioMetaDataEncoder, OggVorbis, \
    OggVorbisDecoder, OggVorbisEncoder, OggVorbisFeature, OggVorbisWrapper, \
    AudioStream, Resampler, ChunkSizeBytes

from spectral import \
    SlidingWindow, OggVorbisWindowingFunc, WindowingFunc, \
    FFT, MDCT, DCT, DCTIV, BarkBands, Chroma, BFCC, SpectralCentroid, \
    SpectralFlatness, AWeighting, LinearScale, LogScale, FrequencyBand, \
    FrequencyScale, FrequencyDimension, GeometricScale, HanningWindowingFunc, \
    FrequencyAdaptiveTransform, ExplicitScale, ExplicitFrequencyDimension, \
    FrequencyAdaptive, FrequencyWeighting

from loudness import \
    log_modulus, inverse_log_modulus, decibel, mu_law, MuLaw, LogModulus, \
    inverse_mu_law

from segment import MeasureOfTransience, MovingAveragePeakPicker, \
    ComplexDomain, TimeSliceFeature

from synthesize import \
    FFTSynthesizer, DCTSynthesizer, TickSynthesizer, NoiseSynthesizer, \
    SineSynthesizer, DCTIVSynthesizer, MDCTSynthesizer, \
    FrequencyAdaptiveFFTSynthesizer, FrequencyAdaptiveDCTSynthesizer, \
    SilenceSynthesizer, WindowedAudioSynthesizer

from learn import \
    KMeans, Learned, MeanStdNormalization, UnitNorm, Log, Multiply, \
    PreprocessingPipeline, Slicer, ReservoirSampler, simple_settings, \
    SklearnModel, WithComponents, InstanceScaling, Reshape, ShuffledSamples, \
    PyTorchNetwork, PyTorchGan, PyTorchAutoEncoder, GanTrainer, \
    WassersteinGanTrainer, SupervisedTrainer, TripletEmbeddingTrainer, \
    Weighted, MuLawCompressed, SimHash, AbsoluteValue, Binarize, Sharpen, \
    learning_pipeline

from ui import ZoundsApp, ZoundsSearch, RangeUnitUnsupportedException

from index import \
    SearchResults, HammingDb, HammingIndex, BruteForceSearch, \
    HammingDistanceBruteForceSearch

from basic import \
    Slice, Sum, Max, Pooled, process_dir, stft, audio_graph, with_onsets, \
    resampled, frequency_adaptive

from util import \
    simple_lmdb_settings, simple_in_memory_settings, \
    simple_object_storage_settings

from nputil import sliding_window

from core import IdentityDimension, ArrayWithUnits

from persistence import \
    ArrayWithUnitsFeature, AudioSamplesFeature, FrequencyAdaptiveFeature

from datasets import \
    PhatDrumLoops, InternetArchive, FreeSoundSearch, DataSetCache, Directory, \
    ingest, MusicNet, NSynth