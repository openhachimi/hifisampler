import numpy as np


def slaney_mel_filterbank(
    sr: float,
    n_fft: int,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Compute a Slaney-normalized Mel filterbank matrix identical to librosa.filters.mel."""
    if fmax is None:
        fmax = float(sr) / 2.0
    fmin = float(fmin)
    fmax = float(fmax)
    n_mels = int(n_mels)

    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    min_mel = (
        min_log_mel + np.log(fmin / min_log_hz) / logstep
        if fmin >= min_log_hz
        else fmin / f_sp
    )
    max_mel = (
        min_log_mel + np.log(fmax / min_log_hz) / logstep
        if fmax >= min_log_hz
        else fmax / f_sp
    )

    mels = np.linspace(min_mel, max_mel, n_mels + 2)
    mel_f = f_sp * mels
    log_t = mels >= min_log_mel
    mel_f[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    lower = -ramps[:-2] / fdiff[:-1, np.newaxis]
    upper = ramps[2:] / fdiff[1:, np.newaxis]
    weights = np.maximum(0.0, np.minimum(lower, upper)).astype(np.float32)

    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights


class AttrDict(dict):
    """A dictionary with attribute-style access. It maps attribute access to
    the real dictionary.  """
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __getstate__(self):
        return self.__dict__.items()

    def __setstate__(self, items):
        for key, val in items:
            self.__dict__[key] = val

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, dict.__repr__(self))

    def __setitem__(self, key, value):
        return super(AttrDict, self).__setitem__(key, value)

    def __getitem__(self, name):
        return super(AttrDict, self).__getitem__(name)

    def __delitem__(self, name):
        return super(AttrDict, self).__delitem__(name)

    __getattr__ = __getitem__
    __setattr__ = __setitem__

    def copy(self):
        return AttrDict(self)

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

if __name__ == '__main__':
    # test
    pass