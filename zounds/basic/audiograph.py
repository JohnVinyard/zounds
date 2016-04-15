import numpy as np
from featureflow import BaseModel, JSONFeature, ByteStream, ByteStreamFeature
from zounds.soundfile import \
    MetaData, AudioMetaDataEncoder, OggVorbis, OggVorbisFeature, AudioStream, \
    Resampler
from zounds.segment import \
    ComplexDomain, MovingAveragePeakPicker, TimeSliceFeature
from zounds.timeseries import \
    ConstantRateTimeSeriesFeature, SR44100, HalfLapped, Stride
from zounds.spectral import \
    SlidingWindow, OggVorbisWindowingFunc, FFT, BarkBands, SpectralCentroid, \
    Chroma, BFCC, DCT


def stft(
        chunksize_bytes=2 * 44100 * 30 * 2,
        resample_to=SR44100(),
        wscheme=HalfLapped(),
        store_fft=False):

    class ShortTimeFourierTransform(BaseModel):
        meta = JSONFeature(
                MetaData,
                store=True,
                encoder=AudioMetaDataEncoder)

        raw = ByteStreamFeature(
                ByteStream,
                chunksize=chunksize_bytes,
                needs=meta,
                store=False)

        ogg = OggVorbisFeature(
                OggVorbis,
                needs=raw,
                store=True)

        pcm = ConstantRateTimeSeriesFeature(
                AudioStream,
                needs=raw,
                store=False)

        resampled = ConstantRateTimeSeriesFeature(
                Resampler,
                needs=pcm,
                samplerate=resample_to,
                store=False)

        windowed = ConstantRateTimeSeriesFeature(
                SlidingWindow,
                needs=resampled,
                wscheme=wscheme,
                wfunc=OggVorbisWindowingFunc(),
                store=False)

        fft = ConstantRateTimeSeriesFeature(
                FFT,
                needs=windowed,
                store=store_fft)

    return ShortTimeFourierTransform


def audio_graph(
        chunksize_bytes=2 * 44100 * 30 * 2,
        resample_to=SR44100(),
        freesound_api_key=None,
        store_fft=False):
    """
    Produce a base class suitable as a starting point for many audio processing
    pipelines.  This class resamples all audio to a common sampling rate, and
    produces a bark band spectrogram from overlapping short-time fourier
    transform frames.  It also compresses the audio into ogg vorbis format for
    compact storage.
    """

    class AudioGraph(BaseModel):
        meta = JSONFeature(
                MetaData,
                store=True,
                encoder=AudioMetaDataEncoder)

        raw = ByteStreamFeature(
                ByteStream,
                chunksize=chunksize_bytes,
                needs=meta,
                store=False)

        ogg = OggVorbisFeature(
                OggVorbis,
                needs=raw,
                store=True)

        pcm = ConstantRateTimeSeriesFeature(
                AudioStream,
                needs=raw,
                store=False)

        resampled = ConstantRateTimeSeriesFeature(
                Resampler,
                needs=pcm,
                samplerate=resample_to,
                store=False)

        windowed = ConstantRateTimeSeriesFeature(
                SlidingWindow,
                needs=resampled,
                wscheme=HalfLapped(),
                wfunc=OggVorbisWindowingFunc(),
                store=False)

        dct = ConstantRateTimeSeriesFeature(
                DCT,
                needs=windowed,
                store=True)

        fft = ConstantRateTimeSeriesFeature(
                FFT,
                needs=windowed,
                store=store_fft)

        bark = ConstantRateTimeSeriesFeature(
                BarkBands,
                needs=fft,
                samplerate=resample_to,
                store=True)

        centroid = ConstantRateTimeSeriesFeature(
                SpectralCentroid,
                needs=bark,
                store=True)

        chroma = ConstantRateTimeSeriesFeature(
                Chroma,
                needs=fft,
                samplerate=resample_to,
                store=True)

        bfcc = ConstantRateTimeSeriesFeature(
                BFCC,
                needs=fft,
                store=True)

    return AudioGraph


def with_onsets(fft_feature):
    """
    Produce a mixin class that extracts onsets
    :param fft_feature: The short-time fourier transform feature
    :return: A mixin class that extracts onsets
    """

    class Onsets(BaseModel):
        onset_prep = ConstantRateTimeSeriesFeature(
                SlidingWindow,
                needs=fft_feature,
                wscheme=HalfLapped() * Stride(frequency=1, duration=3),
                store=False)

        complex_domain = ConstantRateTimeSeriesFeature(
                ComplexDomain,
                needs=onset_prep,
                store=False)

        sliding_detection = ConstantRateTimeSeriesFeature(
                SlidingWindow,
                needs=complex_domain,
                wscheme=HalfLapped() * Stride(frequency=1, duration=11),
                padwith=5,
                store=False)

        slices = TimeSliceFeature(
                MovingAveragePeakPicker,
                needs=sliding_detection,
                aggregate=np.median,
                store=True)

    return Onsets
