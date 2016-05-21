__version__ = '0.4.2'

from timeseries import \
    Hours, Minutes, Seconds, Milliseconds, Microseconds, Picoseconds, \
    AudioSamples, AudioSamplesEncoder, GreedyAudioSamplesDecoder, \
    AudioSamplesFeature, \
    SR11025, SR22050, SR44100, SR48000, SR96000, HalfLapped, Stride, \
    TimeSlice, ConstantRateTimeSeriesEncoder, ConstantRateTimeSeriesFeature, \
    GreedyConstantRateTimeSeriesDecoder, PackedConstantRateTimeSeriesEncoder

from soundfile import \
    MetaData, AudioMetaDataEncoder, FreesoundOrgConfig, \
    OggVorbis, OggVorbisDecoder, OggVorbisEncoder, OggVorbisFeature, \
    OggVorbisWrapper, \
    AudioStream, \
    Resampler

from spectral import \
    SlidingWindow, OggVorbisWindowingFunc, \
    FFT, DCT, BarkBands, Chroma, BFCC, SpectralCentroid, SpectralFlatness

from segment import \
    MeasureOfTransience, MovingAveragePeakPicker, SparseTimestampDecoder, \
    SparseTimestampEncoder, TimeSliceDecoder, TimeSliceFeature, ComplexDomain

from synthesize import FFTSynthesizer, DCTSynthesizer, TickSynthesizer

from learn import \
    KMeans, BinaryRbm, LinearRbm, Learned, \
    MeanStdNormalization, UnitNorm, Log, PreprocessingPipeline, \
    ReservoirSampler, \
    TemplateMatch

from ui import ZoundsApp, RangeUnitUnsupportedException

from index import \
    Contiguous, Offsets, HammingDistanceScorer, PackedHammingDistanceScorer, \
    ConstantRateTimeSliceBuilder, VariableRateTimeSliceBuilder, Search, \
    SearchResults

from basic import Slice, Sum, Max, process_dir, stft, audio_graph, with_onsets
