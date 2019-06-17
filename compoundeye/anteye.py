import numpy as np
from .model import CompoundEye


class AntEye(CompoundEye):

    def __init__(self, ommatidia=None, height=10, width=36):

        if ommatidia is None:
            fov = (-np.pi/6, np.pi/3)

            ground = np.abs(fov[0]) / (np.pi / 2)
            sky = np.abs(fov[1]) / (np.pi / 2)

            Z = (sky + ground) / 2

            thetas = np.linspace(fov[1], fov[0], height, endpoint=True)
            phis = np.linspace(np.pi, -np.pi, width, endpoint=False)
            thetas, phis = np.meshgrid(thetas, phis)
            ommatidia = np.array([thetas.flatten(), phis.flatten()]).T

        super(AntEye, self).__init__(
            #Somehow changed at some point - replacing with default desert ant params, taken from DRA
            n=60,
            omega=56,
            rho=5.4
        )
        #Again, presumably existed in old version of model.py, but alas, not anymore :/
        #self._channel_filters.pop("uv")
        #self._channel_filters.pop("b")
        #self.activate_pol_filters(False)
