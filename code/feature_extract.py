# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as sts
from tqdm import tqdm
import os
# from pywt import WaveletPacket
epsn = 1e-8

# =================== STATISTICAL FEATURES IN TIME DOMAIN =====================
def mean_fea(a):
	return np.mean(a)

def rms_fea(a):
	return np.sqrt(np.mean(np.square(a)))

def sr_fea(a):
	return np.square(np.mean(np.sqrt(np.abs(a))))

def am_fea(a):
	return np.mean(np.abs(a))

def skew_fea(a):
	return np.mean((a-mean_fea(a))**3)

def kurt_fea(a):
	return np.mean((a-mean_fea(a))**4)

def max_fea(a):
	return np.max(a)

def min_fea(a):
	return np.min(a)

def pp_fea(a):
	return max_fea(a)-min_fea(a)

def var_fea(a):
	n = len(a)
	return np.sum((a-mean_fea(a))**2)/(n-1)

def waveform_index(a):
	return rms_fea(a)/(am_fea(a)+epsn)

def peak_index(a):
	return max_fea(a)/(rms_fea(a)+epsn)

def impluse_factor(a):
	return max_fea(a)/(am_fea(a)+epsn)

def tolerance_index(a):
	return max_fea(a)/(sr_fea(a)+epsn)

def skew_index(a):
	n = len(a)
	temp1 = np.sum((a-mean_fea(a))**3)
	temp2 = (np.sqrt(var_fea(a)))**3
	return temp1/((n-1)*temp2)

def kurt_index(a):
	n = len(a)
	temp1 = np.sum((a-mean_fea(a))**4)
	temp2 = (np.sqrt(var_fea(a)))**4
	return temp1/((n-1)*temp2)
# ============================= END ======================================
# def wave_fea(a):
# 	wp = WaveletPacket(a,'db1', maxlevel=8)
# 	nodes = wp.get_level(8, "freq")
# 	return np.linalg.norm(np.array([n.data for n in nodes]), 2)

# =============== STATISTICAL FEATURES IN TIME DOMAIN =======================
# def fft_fft(sequence_data):
# 	fft_trans = np.abs(np.fft.fft(sequence_data))
# 	dc = fft_trans[0]
# 	freq_spectrum = fft_trans[1:int(np.floor(len(sequence_data) * 1.0 / 2)) + 1]
# 	freq_sum_ = np.sum(freq_spectrum)
# 	return dc, freq_spectrum, freq_sum_

def fft_mean(sequence_data):
	def fft_fft(sequence_data):
		fft_trans = np.abs(np.fft.fft(sequence_data))
		# dc = fft_trans[0]
		freq_spectrum = fft_trans[1:int(np.floor(len(sequence_data) * 1.0 / 2)) + 1]
		_freq_sum_ = np.sum(freq_spectrum)
		return freq_spectrum, _freq_sum_
	freq_spectrum, _freq_sum_ = fft_fft(sequence_data)
	return np.mean(freq_spectrum)

def fft_var(sequence_data):
	def fft_fft(sequence_data):
		fft_trans = np.abs(np.fft.fft(sequence_data))
		# dc = fft_trans[0]
		freq_spectrum = fft_trans[1:int(np.floor(len(sequence_data) * 1.0 / 2)) + 1]
		_freq_sum_ = np.sum(freq_spectrum)
		return freq_spectrum, _freq_sum_
	freq_spectrum, _freq_sum_ = fft_fft(sequence_data)
	return np.var(freq_spectrum)

# def fft_std(sequence_data):
# 	def fft_fft(sequence_data):
# 		fft_trans = np.abs(np.fft.fft(sequence_data))
# 		# dc = fft_trans[0]
# 		freq_spectrum = fft_trans[1:int(np.floor(len(sequence_data) * 1.0 / 2)) + 1]
# 		_freq_sum_ = np.sum(freq_spectrum)
# 		return freq_spectrum, _freq_sum_
# 	freq_spectrum, _freq_sum_ = fft_fft(sequence_data)
# 	return np.std(freq_spectrum)

def fft_entropy(sequence_data):
	def fft_fft(sequence_data):
		fft_trans = np.abs(np.fft.fft(sequence_data))
		# dc = fft_trans[0]
		freq_spectrum = fft_trans[1:int(np.floor(len(sequence_data) * 1.0 / 2)) + 1]
		_freq_sum_ = np.sum(freq_spectrum)
		return freq_spectrum, _freq_sum_
	freq_spectrum, _freq_sum_ = fft_fft(sequence_data)
	pr_freq = freq_spectrum * 1.0 / _freq_sum_
	entropy = -1 * np.sum([np.log2(p+1e-5) * p for p in pr_freq])
	return entropy

def fft_energy(sequence_data):
	def fft_fft(sequence_data):
		fft_trans = np.abs(np.fft.fft(sequence_data))
		# dc = fft_trans[0]
		freq_spectrum = fft_trans[1:int(np.floor(len(sequence_data) * 1.0 / 2)) + 1]
		_freq_sum_ = np.sum(freq_spectrum)
		return freq_spectrum, _freq_sum_
	freq_spectrum, _freq_sum_ = fft_fft(sequence_data)
	return np.sum(freq_spectrum ** 2) / len(freq_spectrum)

def fft_skew(sequence_data):
	def fft_fft(sequence_data):
		fft_trans = np.abs(np.fft.fft(sequence_data))
		# dc = fft_trans[0]
		freq_spectrum = fft_trans[1:int(np.floor(len(sequence_data) * 1.0 / 2)) + 1]
		_freq_sum_ = np.sum(freq_spectrum)
		return freq_spectrum, _freq_sum_
	freq_spectrum, _freq_sum_ = fft_fft(sequence_data)
	_fft_mean, _fft_std = fft_mean(sequence_data), fft_std(sequence_data)
	return np.mean([0 if _fft_std < epsn else np.power((x - _fft_mean) / _fft_std, 3)
					for x in freq_spectrum])

def fft_kurt(sequence_data):
	def fft_fft(sequence_data):
		fft_trans = np.abs(np.fft.fft(sequence_data))
		# dc = fft_trans[0]
		freq_spectrum = fft_trans[1:int(np.floor(len(sequence_data) * 1.0 / 2)) + 1]
		_freq_sum_ = np.sum(freq_spectrum)
		return freq_spectrum, _freq_sum_
	freq_spectrum, _freq_sum_ = fft_fft(sequence_data)
	_fft_mean, _fft_std = fft_mean(sequence_data), fft_std(sequence_data)
	return np.mean([0 if _fft_std < epsn else np.power((x - _fft_mean) / _fft_std, 4)
					for x in freq_spectrum])

def fft_shape_mean(sequence_data):
	def fft_fft(sequence_data):
		fft_trans = np.abs(np.fft.fft(sequence_data))
		# dc = fft_trans[0]
		freq_spectrum = fft_trans[1:int(np.floor(len(sequence_data) * 1.0 / 2)) + 1]
		_freq_sum_ = np.sum(freq_spectrum)
		return freq_spectrum, _freq_sum_
	freq_spectrum, _freq_sum_ = fft_fft(sequence_data)
	shape_sum = np.sum([x * freq_spectrum[x]
						for x in range(len(freq_spectrum))])
	return 0 if _freq_sum_ < epsn else shape_sum * 1.0 / _freq_sum_

def fft_shape_std(sequence_data):
	def fft_fft(sequence_data):
		fft_trans = np.abs(np.fft.fft(sequence_data))
		# dc = fft_trans[0]
		freq_spectrum = fft_trans[1:int(np.floor(len(sequence_data) * 1.0 / 2)) + 1]
		_freq_sum_ = np.sum(freq_spectrum)
		return freq_spectrum, _freq_sum_
	freq_spectrum, _freq_sum_ = fft_fft(sequence_data)
	shape_mean = fft_shape_mean(sequence_data)
	var = np.sum([0 if _freq_sum_ < epsn else np.power((x - shape_mean), 2) * freq_spectrum[x]
				  for x in range(len(freq_spectrum))]) / _freq_sum_
	return np.sqrt(var)

def fft_shape_skew(sequence_data):
	def fft_fft(sequence_data):
		fft_trans = np.abs(np.fft.fft(sequence_data))
		# dc = fft_trans[0]
		freq_spectrum = fft_trans[1:int(np.floor(len(sequence_data) * 1.0 / 2)) + 1]
		_freq_sum_ = np.sum(freq_spectrum)
		return freq_spectrum, _freq_sum_
	freq_spectrum, _freq_sum_ = fft_fft(sequence_data)
	shape_mean = fft_shape_mean(sequence_data)
	return np.sum([np.power((x - shape_mean), 3) * freq_spectrum[x]
				   for x in range(len(freq_spectrum))]) / _freq_sum_

def fft_shape_kurt(sequence_data):
	def fft_fft(sequence_data):
		fft_trans = np.abs(np.fft.fft(sequence_data))
		# dc = fft_trans[0]
		freq_spectrum = fft_trans[1:int(np.floor(len(sequence_data) * 1.0 / 2)) + 1]
		_freq_sum_ = np.sum(freq_spectrum)
		return freq_spectrum, _freq_sum_
	freq_spectrum, _freq_sum_ = fft_fft(sequence_data)
	shape_mean = fft_shape_mean(sequence_data)
	return np.sum([np.power((x - shape_mean), 4) * freq_spectrum[x] - 3
				   for x in range(len(freq_spectrum))]) / _freq_sum_
# =================================== END =====================================