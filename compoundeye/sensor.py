from .model import DRA, spectrum, CompoundEye
from code.compass import encode_sph, decode_sph
from .geometry import fibonacci_sphere, angles_distribution, LENS_RADIUS, A_lens
import numpy as np
import os

__dir__ = os.path.dirname(os.path.realpath(__file__)) + "/"
__datadir__ = __dir__ + "../data/sensor/"

#np.random.seed(2018)
NB_EN = 8
DEBUG = False

class POLCompassDRA(DRA):

    def __init__(self, n=60, omega=56, rho=5.4, nb_pr=2):
        super(POLCompassDRA, self).__init__(n=n, omega=omega, rho=rho, nb_pr=nb_pr, name="pol")
        self.rhabdom = np.array([[spectrum["uv"], spectrum["uv"]]] * n).T
        self.mic_l = np.array([[0., np.pi / 2]] * n).T


class CompassSensor(CompoundEye):
    def __init__(self, nb_lenses=60, fov=np.deg2rad(60), thetas=None, phis=None, alphas=None,
                 kernel=None, mode="cross", fibonacci=False):
        self.ans = 42
    #Dummy class;
    #real version removed from here and impossible to re-add, but other, currently unused code for cx expects it


def encode_sun(lon, lat):
    return encode_sph(lat, lon, length=NB_EN)


def decode_sun(x):
    lat, lon = decode_sph(x)
    return lon, lat



if __name__ == "__main__":
    from environment import Sky
    from model import visualise

    sky = Sky(theta_s=np.pi/3)
    dra = POLCompassDRA()
    dra.theta_t = np.pi/6
    dra.phi_t = np.pi/3
    # s = dra(sky)
    r_pol = dra(sky)
    r_po = dra.r_po
    # print (s.shape)

    visualise(sky, r_po)
