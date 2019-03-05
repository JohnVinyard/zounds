import numpy as np


class Hours(np.timedelta64):
    """
    Convenience class for creating a duration in hours

    Args:
        hours (int): duration in hours

    Examples:
        >>> from zounds import Hours
        >>> hours = Hours(3)
        >>> hours
        numpy.timedelta(3, 'h')
    """
    def __new__(cls, hours):
        return np.timedelta64(int(hours), 'h')


class Minutes(np.timedelta64):
    """
    Convenience class for creating a duration in minutes

    Args:
        minutes (int): duration in minutes

    Examples:
        >>> from zounds import Minutes
        >>> minutes = Minutes(3)
        >>> minutes
        numpy.timedelta(3, 'm')
    """
    def __new__(cls, minutes):
        return np.timedelta64(int(minutes), 'm')


class Seconds(np.timedelta64):
    """
    Convenience class for creating a duration in seconds

    Args:
        seconds (int): duration in seconds

    Examples:
        >>> from zounds import Seconds
        >>> seconds = Seconds(3)
        >>> seconds
        numpy.timedelta(3, 's')
    """
    def __new__(cls, seconds):
        return np.timedelta64(int(seconds), 's')


class Milliseconds(np.timedelta64):
    """
    Convenience class for creating a duration in milliseconds

    Args:
        milliseconds (int): duration in milliseconds

    Examples:
        >>> from zounds import Milliseconds
        >>> ms = Milliseconds(3)
        >>> ms
        numpy.timedelta(3, 'ms')
    """
    def __new__(cls, milliseconds):
        return np.timedelta64(int(milliseconds), 'ms')


class Microseconds(np.timedelta64):
    """
    Convenience class for creating a duration in microseconds

    Args:
        microseconds (int): duration in microseconds

    Examples:
        >>> from zounds import Microseconds
        >>> us = Microseconds(3)
        >>> us
        numpy.timedelta(3, 'us')
    """
    def __new__(cls, microseconds):
        return np.timedelta64(int(microseconds), 'us')


class Nanoseconds(np.timedelta64):
    """
    Convenience class for creating a duration in nanoseconds

    Args:
        nanoseconds (int): duration in nanoseconds

    Examples:
        >>> from zounds import Nanoseconds
        >>> ns = Nanoseconds(3)
        >>> ns
        numpy.timedelta(3, 'ns')
    """
    def __new__(cls, nanoseconds):
        return np.timedelta64(int(nanoseconds), 'ns')


class Picoseconds(np.timedelta64):
    """
    Convenience class for creating a duration in picoseconds

    Args:
        picoseconds (int): duration in picoseconds

    Examples:
        >>> from zounds import Picoseconds
        >>> ps = Picoseconds(3)
        >>> ps
        numpy.timedelta(3, 'ps')
    """
    def __new__(cls, picoseconds):
        return np.timedelta64(int(picoseconds), 'ps')
