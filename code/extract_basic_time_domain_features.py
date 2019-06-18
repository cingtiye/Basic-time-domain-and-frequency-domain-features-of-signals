# -*- coding: utf-8 -*-

import pandas as pd
import scipy.stats
import numpy as np
from sklearn.metrics import mean_squared_error


class ExtractTimeFeatures:
    def __init__(self, z, name='z'):
        if type(z) is type(pd.DataFrame([1])):
            z = z.values
        if type(z) is not type(np.array([1])):
            try:
                z = np.array(z)
            except TypeError:
                print("The type of z must be 'numpy.ndarray' or 'pandas.core.frame.DataFrame'")

        self.c = []  # all feature names
        self.z = z  # input dataset whose type is 'np.array'
        self.name = '_' + name  # dataset name
        self._df = []  # return new dataset which includes every feature name

    def extracttimefeatures(self):
        max_z = self.get_max()
        min_z = self.get_min()
        range_z = self.get_range()
        mean_z = self.get_mean()
        median_z = self.get_median()
        mode_z = self.get_mode()
        std_z = self.get_std()
        rms_z = self.get_rms()
        ms_z = self.get_ms()
        k_center_z = self.get_k_order_moment(is_center=True)
        k_origin_z = self.get_k_order_moment(is_origin=True)
        skew_z = self.get_skew()
        kurt_z = self.get_kurt()
        kurt_factor_z = self.get_kurt_factor()
        wave_factor = self.get_wave_factor()
        pulse_factor_z = self.get_pulse_factor()
        margin_factor_z = self.get_margin_factor()

        self._df = pd.DataFrame([
            max_z, min_z, range_z, mean_z, median_z, mode_z, std_z,
            rms_z, ms_z, k_center_z, k_origin_z, skew_z, kurt_z,
            kurt_factor_z, wave_factor, pulse_factor_z, margin_factor_z
        ], self.c).transpose()
        return self._df

# ==============Dimensional time domain feature=====================
    def get_max(self):
        self.max_z = np.max(self.z, axis=1)
        self.c.append('max'+self.name)
        return self.max_z

    def get_min(self):
        self.min_z = np.min(self.z, axis=1)
        self.c.append('min' + self.name)
        return self.min_z

    def get_range(self):
        self.range_z = self.max_z-self.min_z
        self.c.append('range' + self.name)
        return self.range_z

    def get_mean(self):
        self.mean_z = np.mean(self.z, axis=1)
        self.c.append('mean' + self.name)
        return self.mean_z

    def get_median(self):
        self.median_z = np.median(self.z, axis=1)
        self.c.append('median' + self.name)
        return self.median_z

    def get_mode(self):
        self.mode_z = scipy.stats.mode(self.z, axis=1)[0].reshape([-1])
        self.c.append('mode' + self.name)
        return self.mode_z

    def get_std(self):
        self.std_z = np.std(self.z ,axis=1)
        self.c.append('std' + self.name)
        return self.std_z

    def get_rms(self):
        rms_z = [np.sqrt(mean_squared_error(zi, np.zeros(len(zi)))) for zi in self.z]
        self.rms_z = np.array(rms_z)
        self.c.append('rms' + self.name)
        return self.rms_z

    def get_ms(self):
        ms_z = [mean_squared_error(zi, np.zeros(len(zi))) for zi in self.z]
        self.ms_z = np.array(ms_z)
        self.c.append('ms' + self.name)
        return self.ms_z

    def get_k_order_moment(self, k=3, is_center=False, is_origin=False):
        moment_name, self.moment_z = self.k_order_moment(self.z, k,
                                                         is_center, is_origin)
        self.c.append(moment_name + self.name)
        return self.moment_z

    @staticmethod
    def k_order_moment(z, k, is_center, is_origin):
        """
        Calculate k-order center moment and k-order origin moment of z
        :param z: array_like
        :param k: int
        :param is_center: bool; whether calculate k-order center moment
        :param is_origin: bool; whether calculate k-order origin moment
        :return: tuple; return k-order center moment or k-order origin moment
        """
        if (is_center is False) and (is_origin is False):
            raise ValueError("At least one of is_center and is_origin is True")
        if (is_center is True) and (is_origin is True):
            raise ValueError("At most one of is_center and is_origin is True")
        if (type(k) is not int) or (k < 0):
            raise TypeError("k must be a integrate and more than 0")
        if type(z) is list:
            z = np.array(z)

        mean_z = np.mean(z, axis=1)
        if is_origin is False:
            k_center = np.mean([(z[i] - mean_z[i]) ** k for i in range(z.shape[0])], axis=1)
            return str(k)+'_order_center', k_center
        if is_center is False:
            k_origin = np.mean([z[i] ** k for i in range(z.shape[0])], axis=1)
            return str(k)+'_order_origin', k_origin
# =========================END=========================================

# ===============Dimensionless time domain feature=====================
    def get_skew(self):
        self.skew_z = pd.DataFrame(self.z.transpose()).skew().values
        self.c.append('skew' + self.name)
        return self.skew_z

    def get_kurt(self):
        self.kurt_z = pd.DataFrame(self.z.transpose()).kurt().values
        self.c.append('kurt' + self.name)
        return self.kurt_z

    def get_kurt_factor(self):
        self.kurt_factor_z = self.max_z/self.rms_z
        self.c.append('kurt_factor' + self.name)
        return self.kurt_factor_z

    def get_wave_factor(self):
        self.wave_factor_z = self.rms_z/self.mean_z
        self.c.append('wave_factor' + self.name)
        return self.wave_factor_z

    def get_pulse_factor(self):
        self.pulse_factor_z = self.max_z/abs(self.mean_z)
        self.c.append('pulse_factor' + self.name)
        return self.pulse_factor_z

    def get_margin_factor(self):
        self.margin_factor_z = self.max_z/self.ms_z
        self.c.append('margin_factor' + self.name)
        return self.margin_factor_z
# =========================END=========================================


if __name__ == '__main__':
    # =====Example=======
    z_ = [[.1, 2, .3, 4, 5, .6, 7], [.1, 12, .3, 41, 15, .6, .7]]
    c = ExtractTimeFeatures(z_)
    print(c.extracttimefeatures())
