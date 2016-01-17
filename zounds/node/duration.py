import numpy as np


class Hours(np.timedelta64):
    def __new__(cls, hours):
        return np.timedelta64(hours, 'h')


class Minutes(np.timedelta64):
    def __new__(cls, minutes):
        return np.timedelta64(minutes, 'm')


class Seconds(np.timedelta64):
    def __new__(cls, seconds):
        return np.timedelta64(seconds, 's')


class Milliseconds(np.timedelta64):
    def __new__(cls, milliseconds):
        return np.timedelta64(milliseconds, 'ms')


class Microseconds(np.timedelta64):
    def __new__(cls, microseconds):
        return np.timedelta64(microseconds, 'us')


class Nanoseconds(np.timedelta64):
    def __new__(cls, nanoseconds):
        return np.timedelta64(nanoseconds, 'ns')


class Picoseconds(np.timedelta64):
    def __new__(cls, picoseconds):
        return np.timedelta64(picoseconds, 'ps')
