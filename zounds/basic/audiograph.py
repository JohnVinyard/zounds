import numpy as np
from featureflow import BaseModel, JSONFeature, ByteStream, ByteStreamFeature
from zounds.soundfile import \
    MetaData, AudioMetaDataEncoder, OggVorbis, OggVorbisFeature, AudioStream, \
    Resampler, ChunkSizeBytes
from zounds.segment import \
    ComplexDomain, MovingAveragePeakPicker, TimeSliceFeature
from zounds.persistence import ArrayWithUnitsFeature, AudioSamplesFeature, \
    FrequencyAdaptiveFeature
from zounds.timeseries import SR44100, HalfLapped, Stride, Seconds
from zounds.spectral import \
    SlidingWindow, OggVorbisWindowingFunc, FFT, BarkBands, SpectralCentroid, \
    Chroma, BFCC, DCT, FrequencyAdaptiveTransform

DEFAULT_CHUNK_SIZE = ChunkSizeBytes(
    samplerate=SR44100(),
    duration=Seconds(30),
    bit_depth=16,
    channels=2)


def resampled(
        chunksize_bytes=DEFAULT_CHUNK_SIZE,
        resample_to=SR44100(),
        store_resampled=False):
    """
    Create a basic processing pipeline that can resample all incoming audio
    to a normalized sampling rate for downstream processing, and store a
    convenient, compressed version for playback

    :param chunksize_bytes: The number of bytes from the raw stream to process
    at once
    :param resample_to: The new, normalized sampling rate
    :return: A simple processing pipeline
    """

    class Resampled(BaseModel):
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

        pcm = AudioSamplesFeature(
            AudioStream,
            needs=raw,
            store=False)

        resampled = AudioSamplesFeature(
            Resampler,
            needs=pcm,
            samplerate=resample_to,
            store=store_resampled)

    return Resampled


def frequency_adaptive(
        long_window_sample_rate,
        scale,
        store_freq_adaptive=False,
        check_scale_overlap_ratio=False,
        chunksize_bytes=DEFAULT_CHUNK_SIZE,
        resample_to=SR44100(),
        store_resampled=False):

    BaseModel = resampled(chunksize_bytes, resample_to, store_resampled)

    class FrequencyAdaptive(BaseModel):
        long_windowed = ArrayWithUnitsFeature(
            SlidingWindow,
            wscheme=long_window_sample_rate,
            wfunc=OggVorbisWindowingFunc(),
            needs=BaseModel.resampled,
            store=False)

        long_fft = ArrayWithUnitsFeature(
            FFT,
            needs=long_windowed,
            store=False)

        freq_adaptive = FrequencyAdaptiveFeature(
            FrequencyAdaptiveTransform,
            transform=np.fft.irfft,
            scale=scale,
            check_scale_overlap_ratio=check_scale_overlap_ratio,
            window_func=np.hanning,
            needs=long_fft,
            store=store_freq_adaptive)

    return FrequencyAdaptive


def stft(
        chunksize_bytes=DEFAULT_CHUNK_SIZE,
        resample_to=SR44100(),
        wscheme=HalfLapped(),
        store_fft=False,
        fft_padding_samples=None,
        store_windowed=False,
        store_resampled=False):

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

        pcm = AudioSamplesFeature(
            AudioStream,
            needs=raw,
            store=False)

        resampled = AudioSamplesFeature(
            Resampler,
            needs=pcm,
            samplerate=resample_to,
            store=store_resampled)

        windowed = ArrayWithUnitsFeature(
            SlidingWindow,
            needs=resampled,
            wscheme=wscheme,
            wfunc=OggVorbisWindowingFunc(),
            store=store_windowed)

        fft = ArrayWithUnitsFeature(
            FFT,
            padding_samples=fft_padding_samples,
            needs=windowed,
            store=store_fft)

    return ShortTimeFourierTransform


def audio_graph(
        chunksize_bytes=DEFAULT_CHUNK_SIZE,
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

        pcm = AudioSamplesFeature(
            AudioStream,
            needs=raw,
            store=False)

        resampled = AudioSamplesFeature(
            Resampler,
            needs=pcm,
            samplerate=resample_to,
            store=False)

        windowed = ArrayWithUnitsFeature(
            SlidingWindow,
            needs=resampled,
            wscheme=HalfLapped(),
            wfunc=OggVorbisWindowingFunc(),
            store=False)

        dct = ArrayWithUnitsFeature(
            DCT,
            needs=windowed,
            store=True)

        fft = ArrayWithUnitsFeature(
            FFT,
            needs=windowed,
            store=store_fft)

        bark = ArrayWithUnitsFeature(
            BarkBands,
            needs=fft,
            samplerate=resample_to,
            store=True)

        centroid = ArrayWithUnitsFeature(
            SpectralCentroid,
            needs=bark,
            store=True)

        chroma = ArrayWithUnitsFeature(
            Chroma,
            needs=fft,
            samplerate=resample_to,
            store=True)

        bfcc = ArrayWithUnitsFeature(
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
        onset_prep = ArrayWithUnitsFeature(
            SlidingWindow,
            needs=fft_feature,
            wscheme=HalfLapped() * Stride(frequency=1, duration=3),
            store=False)

        complex_domain = ArrayWithUnitsFeature(
            ComplexDomain,
            needs=onset_prep,
            store=False)

        sliding_detection = ArrayWithUnitsFeature(
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
