from os.path import join, isdir
from os import listdir
import os
import struct
import numpy as np
from scipy import sparse, signal
import matplotlib.pyplot as plt
from scipy.optimize import fmin, curve_fit
import socket
import time
import datetime
import shutil


def default_processing(s, solvent = 'MeCN'):
    s.fft()
    s.gen_x_scale()
    #s.autophase()
    s.phase(30, 0)
    if solvent.lower() == 'mecn':
        s.reference(1.96)
    elif solvent.lower() == 'dmso':
        s.reference(2.50)
    elif solvent.lower() == "cdcl3":
        s.reference(7.26)
    elif solvent.lower() == "dioxane":
        s.reference(3.60)
    elif solvent.lower() == 'methanol':
        s.reference(3.34)
    s.normalize()
    s.cut(2.5, 12)
#

class NMRSpectrum():
    def __init__(self, path=None, verbose=False, X_scale=[], spectrum=[]):
        # self.nmr_dir = nmr_dir
        self.verbose = verbose
        self.spectrum = spectrum
        self.X_scale = X_scale
        self.nmr_folder = path

        if path != None:
            # read spectrum parameters
            spectrum_parameters = open(join(path, 'acqu.par'), 'r')
            parameters = spectrum_parameters.readlines()
            self.param_dict = {}
            for param in parameters:
                self.param_dict[param.split('= ')[0].strip(' ')] = \
                    param.split('= ')[1].strip('\n')
            if self.verbose:
                print(self.param_dict)

            # open file with nmr data
            spectrum_path = join(path, 'data.1d')
            # open binary file with spectrum
            nmr_data = open(spectrum_path, mode='rb')
            # read first eight bytes
            spectrum = []
            # unpack the data
            while True:
                data = nmr_data.read(4)
                if not data:
                    break
                spectrum.append(struct.unpack('<f', data))
            # remove fisrt eight points and divide data into three parts
            lenght = int(len(spectrum[8:]) / 3)
            # print (type(spectrum))
            fid = spectrum[lenght + 8:]
            self.gamma = 1 / max(spectrum[8:lenght])[0]
            fid_real = []
            fid_img = []
            for i in range(int(len(fid) / 2)):
                fid_real.append(fid[2 * i][0])
                fid_img.append(fid[2 * i + 1][0])
            self.fid_complex = []
            for i in range(len(fid_real)):
                self.fid_complex.append(np.complex(fid_real[i], fid_img[i] * -1))

    def fft(self):
        self.spectrum = np.fft.fft(self.fid_complex,
                                   n=1 * len(self.fid_complex))
        self.spectrum = np.fft.fftshift(self.spectrum)
        self.spectrum_length = len(self.spectrum)

    def phase(self, phase0=0.0, phase1=0.0):
        phase0_rad = np.pi * phase0 / 180
        phase1_rad = np.pi * phase1 / 180

        phased_spectrum = []
        # phase spectrum
        for (i, point) in enumerate(self.spectrum):
            correction = i * phase1_rad / self.spectrum_length + phase0_rad
            real_part = np.cos(correction) * point.real - \
                        np.sin(correction) * point.imag
            imag_part = np.cos(correction) * point.imag - \
                        np.sin(correction) * point.real
            phased_spectrum.append(np.complex(real_part, imag_part))
        self.spectrum = phased_spectrum

    def show(self):
        plt.xlabel('d [ppm]')
        plt.ylabel('Intensity a.u.')
        plt.plot(self.X_scale, self.spectrum)
        plt.gca().invert_xaxis()
        # if hasattr(self, 'peaks'):
        # plt.plot([i[0] for i in self.peaks],
        # [i[1] for i in self.peaks], 'ro')
        plt.show()

    def integrate(self, low_ppm, high_ppm):
        peak = [self.spectrum[i].real for i, v in
                enumerate(self.X_scale) if (v > low_ppm and v < high_ppm)]
        return np.trapz(peak)

    def gen_x_scale(self):
        self.X_scale = []
        for (i, point) in enumerate(self.spectrum):
            x = (i * 5000. / self.spectrum_length +
                 float(self.param_dict['lowestFrequency'])) / \
                float(self.param_dict['b1Freq'])
            self.X_scale.append(x)

    def cut(self, low_ppm=5, high_ppm=12):
        self.spectrum = [self.spectrum[i] for (i, p) in
                         enumerate(self.X_scale) if (p > low_ppm) and (p < high_ppm)]
        self.X_scale = [i for i in self.X_scale if (i > low_ppm) and (i < high_ppm)]

    def autophase(self):
        def entropy(phase):
            def penalty_function(Ri):
                if Ri >= 0:
                    return 0
                else:
                    return np.square(Ri)

            phase0_rad = np.pi * phase[0] / 180
            phase1_rad = np.pi * phase[1] / 180
            real_spectrum = [i.real for i in self.spectrum]
            p_real_spectrum = []
            for (i, ri) in enumerate(self.spectrum):
                correction = phase0_rad + i * phase1_rad / self.spectrum_length
                ci = ri.real * np.cos(correction) - ri.imag * np.sin(correction)
                p_real_spectrum.append(ci)

            penalty = 1e-14 * np.sum([penalty_function(i) for i in p_real_spectrum])
            first_derivative = np.gradient(p_real_spectrum)
            ssum = np.sum(np.absolute(first_derivative))
            prob = [np.absolute(i) / ssum for i in first_derivative]
            entropy = np.sum([-p * np.log(p) for p in prob])

            # print entropy, penalty
            return entropy + penalty

        new_phase = fmin(entropy, [0, 0], )
        self.phase(new_phase[0], new_phase[1])

    def baseline_als(self, lam, p, niter=10):
        self.spectrum = [i.real for i in self.spectrum]
        L = len(self.spectrum)
        D = sparse.csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = sparse.linalg.spsolve(Z, w * self.spectrum)
            w = p * (self.spectrum > z) + (1 - p) * (self.spectrum < z)
        self.spectrum = self.spectrum - z

    def smooth(self, lam, p, niter=10):
        self.spectrum = [i.real for i in self.spectrum]
        L = len(self.spectrum)
        D = sparse.csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = sparse.linalg.spsolve(Z, w * self.spectrum)
            w = p * (self.spectrum > z) + (1 - p) * (self.spectrum < z)
        self.spectrum = z
        plt.plot(self.X_scale, self.spectrum)
        plt.plot(self.X_scale, z)
        plt.show()
        self.spectrum = z


    def find_peaks(self, thresh=0.3, min_dist=0.5):
        thresh *= (np.max(self.spectrum) - np.min(self.spectrum))
        smooth = signal.savgol_filter(self.spectrum, 3, 2)
        dy = np.diff(smooth)

        peaks = np.where((np.hstack([dy, 0.]) < 0) &
                         (np.hstack([0., dy]) > 0) &
                         (self.spectrum > thresh))[0]

        # peaks = signal.find_peaks_cwt(self.spectrum,
        # np.arange(70,90), noise_perc = 20)
        self.spectrum = np.real(self.spectrum)

        self.peaks = [[self.X_scale[i], self.spectrum[i]] for i in peaks]
        # print self.peaks
        # remove peaks to close to each other
        idx_to_remove = []
        for i, p in enumerate(self.peaks):
            if i not in idx_to_remove:
                for j in range(i + 1, len(self.peaks)):
                    if abs(self.peaks[i][0] - self.peaks[j][0]) < min_dist:
                        if self.peaks[j][1] - self.peaks[i][1] > 0:
                            if i not in idx_to_remove:
                                idx_to_remove.append(i)
                        else:
                            if j not in idx_to_remove:
                                idx_to_remove.append(j)

        sorted_peaks = []
        for i, p in enumerate(self.peaks):
            if i not in idx_to_remove:
                sorted_peaks.append(self.peaks[i])
        self.peaks = sorted_peaks

        #        plt.plot(self.X_scale, self.spectrum)
        #        for p in self.peaks:
        #            params = self.fit_lorentzian(p[0], p[1])
        #            fitted = [self.lorentzian(i, params[0],
        #            params[1], params[2]) for i in self.X_scale]
        #            plt.plot(self.X_scale, fitted)
        #        plt.show()
        return self.peaks

    def lorentzian(self, p, p0, ampl, w):
        x = (p0 - p) / (w / 2)
        return ampl / (1 + x * x)

    def fit_lorentzian(self, x, y):
        initial = [x, y, 0.07]
        params, pcov = curve_fit(self.lorentzian, self.X_scale,
                                 self.spectrum, initial, maxfev=5000)
        return params

    def autointegrate(self, lowb, highb, show=False):
        spectrum = self.spectrum[:]
        X_scale = self.X_scale[:]
        self.cut(lowb, highb)
        # self.smooth(1e1, .9)
        # find peak having the highest intensity
        peak = sorted(self.find_peaks(thresh=0.2), key=lambda x: x[1], reverse=True)[0]
        fit_params = self.fit_lorentzian(peak[0], peak[1])
        fitted = [self.lorentzian(i, fit_params[0], fit_params[1], fit_params[2]) for i in self.X_scale]
        area = np.trapz(fitted)
        # if peak has a half width larger then 0.2 ppm return 0
        # fitting process probably failed
        if show == True:
            plt.plot(self.X_scale, self.spectrum)
            plt.plot(self.X_scale, fitted)
            plt.show()

        self.spectrum = spectrum[:]
        self.X_scale = X_scale[:]

        if fit_params[2] > 0.2:
            return 0
        else:
            return area

    def normalize(self):
        self.spectrum = [i.real for i in self.spectrum]
        max_intensity = sorted(self.spectrum)[-1]
        self.spectrum = [i / max_intensity for i in self.spectrum]

    def __sub__(self, other):
        from operator import sub
        new_spectrum = map(sub, self.spectrum, other.spectrum)
        return nmr_spectrum(X_scale=self.X_scale, spectrum=new_spectrum)

    def __add__(self, other):
        from operator import add
        new_spectrum = map(add, self.spectrum, other.spectrum)
        return nmr_spectrum(X_scale=self.X_scale, spectrum=new_spectrum)

    def __mul__(self, other):
        new_spectrum = [i * other for i in self.spectrum]
        return nmr_spectrum(X_scale=self.X_scale, spectrum=new_spectrum)

    def __rmul__(self, other):
        new_spectrum = [i * other for i in self.spectrum]
        return nmr_spectrum(X_scale=self.X_scale, spectrum=new_spectrum)

    def reference(self, solvent_shift):
        diff = solvent_shift - self.find_peaks(thresh=0.5)[0][0]
        self.X_scale = [i + diff for i in self.X_scale]


if __name__ == '__main__':

    spectrum = NMRSpectrum('1H')
    default_processing(spectrum)
    spectrum.show()


