__version__ = '0.20.9'

from timeseries import \
    Hours, Minutes, Seconds, Milliseconds, Microseconds, Picoseconds, \
    SR11025, SR22050, SR44100, SR48000, SR96000, HalfLapped, Stride, \
    TimeSlice, VariableRateTimeSeries, VariableRateTimeSeriesFeature, \
    SampleRate, AudioSamples

from soundfile import \
    MetaData, AudioMetaDataEncoder, FreesoundOrgConfig, \
    OggVorbis, OggVorbisDecoder, OggVorbisEncoder, OggVorbisFeature, \
    OggVorbisWrapper, \
    AudioStream, \
    Resampler

from spectral import \
    SlidingWindow, OggVorbisWindowingFunc, \
    FFT, MDCT, DCT, DCTIV, BarkBands, Chroma, BFCC, SpectralCentroid, \
    SpectralFlatness, AWeighting, LinearScale, LogScale, FrequencyBand, \
    FrequencyScale

from segment import \
    MeasureOfTransience, MovingAveragePeakPicker, SparseTimestampDecoder, \
    SparseTimestampEncoder, TimeSliceDecoder, TimeSliceFeature, ComplexDomain

from synthesize import \
    FFTSynthesizer, DCTSynthesizer, TickSynthesizer, NoiseSynthesizer, \
    SineSynthesizer, DCTIVSynthesizer, MDCTSynthesizer

from learn import \
    KMeans, BinaryRbm, LinearRbm, Learned, \
    MeanStdNormalization, UnitNorm, Log, Multiply, PreprocessingPipeline, \
    Slicer, Flatten, ReservoirSampler, TemplateMatch, simple_settings

from ui import ZoundsApp, ZoundsSearch, RangeUnitUnsupportedException

from index import \
    Contiguous, Offsets, HammingDistanceScorer, PackedHammingDistanceScorer, \
    ConstantRateTimeSliceBuilder, VariableRateTimeSliceBuilder, Search, \
    SearchResults, hamming_index

from basic import \
    Slice, Sum, Max, Pooled, process_dir, stft, audio_graph, with_onsets, resampled

from util import simple_lmdb_settings, simple_in_memory_settings

from nputil import sliding_window

from core import IdentityDimension, ArrayWithUnits

from persistence import ArrayWithUnitsFeature, AudioSamplesFeature
