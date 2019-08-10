import sys
from PIL import Image
sys.path.insert(0, '../../compmodels')
sys.path.insert(0, '..')
from net.base import Network, RNG

import numpy as np
import sklearn.decomposition
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import yaml
import os
import random


# get path of the script
__dir__ = os.path.dirname(os.path.abspath(__file__))
# load parameters
with open(os.path.join(__dir__, 'Adin2016.yaml'), 'rb') as f:
    params = yaml.safe_load(f)

GAIN = params['gain']
LEARNING_RATE = params['learning-rate']
KC_THRESHOLD = params['kc-threshold']


class WillshawNet(Network):

    def __init__(self, learning_rate=LEARNING_RATE, tau=KC_THRESHOLD, nb_channels=1, **kwargs):
        """

        :param learning_rate: the rate with which the weights are changing
        :type learning_rate: float
        :param tau: the threshold after witch a KC is activated
        :type tau: float
        :param nb_channels: number of colour channels that can be interpreted
        :type nb_channels: int
        """
        super(WillshawNet, self).__init__(**kwargs)
        self.learning_rate = learning_rate
        self._tau = tau
        self.nb_channels = nb_channels
        np.random.seed(2019)

        self.nb_pn = params['mushroom-body']['PN'] * nb_channels
        self.nb_kc = params['mushroom-body']['KC'] * nb_channels
        self.nb_en = params['mushroom-body']['EN']

        #self.w_pn2kc = generate_pn2kc_weights(self.nb_pn, self.nb_kc, dtype=self.dtype)
        self.w_pn2kc = generate_random_pn2kc_weights(self.nb_pn, self.nb_kc, dtype = self.dtype)
        self.w_kc2en = np.ones((self.nb_kc, self.nb_en), dtype=self.dtype)
        self.params = [self.w_pn2kc, self.w_kc2en]
        self.n_hist = np.zeros(self.nb_kc)
        self.lambdas = np.full(self.nb_kc, 600)
        self.density_mask = np.ones(self.nb_kc, dtype=self.dtype)
        self.recorded_pns = np.array([])

        self.f_pn = lambda x: np.maximum(self.dtype(x) / self.dtype(255), 0)
        # self.f_pn = lambda x: np.maximum(self.dtype(self.dtype(x) / self.dtype(255) > .5), 0)
        self.f_kc = lambda x: self.dtype(x > tau)
        self.f_kc_dynamic = lambda x: self.dtype(x > 0)
        self.f_en = lambda x: np.maximum(x, 0)
        self.act_kc = np.array([])

        self.pn = np.zeros(self.nb_pn)
        self.kc = np.zeros(self.nb_kc)
        self.en = np.zeros(self.nb_en)
        self.learn_weights = False
        self.update = False
        self.adapt = False

    def reset(self):
        super(WillshawNet, self).reset()

        self.pn = np.zeros(self.nb_pn)
        self.kc = np.zeros(self.nb_kc)
        self.en = np.zeros(self.nb_en)

        self.w_kc2en = np.ones((self.nb_kc, self.nb_en), dtype=self.dtype)

    def __call__(self, *args, **kwargs):
        self.pn, self.kc, self.en = self._fprop(args[0])
        #print(self.pn)
        #print(self.w_pn2kc)
        if self.update:
            self._update(self.kc)
        if self.adapt:
            self._adapt(self.pn)
        return self.en

    def _fprop(self, pn):
        a_pn = self.f_pn(pn)
        #Normalise the input
        a_pn = 42 * a_pn / np.linalg.norm(a_pn)
        '''if (len(self.recorded_pns) == 0):
            self.recorded_pns = a_pn
        else:
            self.recorded_pns = np.stack(self.recorded_pns,a_pn)'''
        kc = a_pn.dot(self.w_pn2kc)
        #Scale KC activations by the local density, ensuring that KCs in areas with higher densities are proportionally
        #   harder to activate, meaning that the % of activated KCs should remain constant
        #kc = kc / self.density_mask
        a_kc = self.f_kc(kc)
        biased_kc = 10*kc - (self._tau * self.density_mask)
        a_kc = self.f_kc_dynamic(biased_kc)
        print(np.count_nonzero(a_kc)/self.nb_kc)
        if not self.adapt and not self.update:
            print(np.nonzero(a_kc))
        en = a_kc.dot(self.w_kc2en)
        a_en = self.f_en(en)
        return a_pn, a_kc, a_en

    def _adapt(self,pn):
        timesteps = 10
        print("adapting")
        #self.lambdas = np.full(self.nb_kc, 500)
        #print(pn)
        #self.n_hist = self.n_hist / (np.exp(-(50-self.lambdas[0]))+1)
        #self.n_hist = self.n_hist / 20
        #print(np.exp(-(50-self.lambdas[0]))+1)
        #print(self.n_hist)
        w_w_fn = np.vectorize(lambda d, lamb: np.exp( -(d ** 2) / (lamb+15)))
        for i in range(timesteps):
            #Differences between each weight set and the input perception
            diffs = np.vstack(pn) - self.w_pn2kc
            dists = np.linalg.norm(diffs,axis=0)
            print(np.min(dists))
            print(np.max(dists))


            pt = w_w_fn(dists,self.lambdas)
            #print(np.max(pt))
            #print(np.min(pt))
            #print((pt>0.15).sum())
            #print(pt)
            pt = pt / pt.sum()
            #print(pt)

            self.n_hist = self.n_hist + 1.0*pt
            g = 1/self.n_hist

            delta_weights = 0.7*pt*g
            #print(delta_weights)
            delta = delta_weights * diffs
            self.w_pn2kc = self.w_pn2kc + delta
            self.w_pn2kc = 42 * self.w_pn2kc / np.linalg.norm(self.w_pn2kc,axis=0)
            #print(self.w_pn2kc)
            if self.lambdas[0]>25:
                self.lambdas = self.lambdas / 1.05
            else:
                self.lambdas = self.lambdas / 1.01
            mindist_index = np.argmin(dists)
            #self.lambdas[mindist_index] = self.lambdas[mindist_index] / 1.1
            #print(mindist_index)
        nr_winners = np.count_nonzero(pt>(1/self.nb_kc))
        print(nr_winners)
        print("LAMBDA")
        print(self.lambdas[0])
        if self.lambdas[0]<15:
            self.density_mask[pt>(1/self.nb_kc)] = self.density_mask[pt>(1/self.nb_kc)] / 16
            self.density_mask[pt>(1/self.nb_kc)] = self.density_mask[pt>(1/self.nb_kc)] + nr_winners
            print(self.density_mask)
        print("HIST")
        print(np.min(self.n_hist))
        print(np.max(self.n_hist))

    def _update(self, kc):
        """
            THE LEARNING RULE:
        ----------------------------

          KC  | KC2EN(t)| KC2EN(t+1)
        ______|_________|___________
           1  |    1    |=>   0
           1  |    0    |=>   0
           0  |    1    |=>   1
           0  |    0    |=>   0

        :param kc: the KC activation
        :return:
        """
        learning_rule = (kc >= self.w_kc2en[:, 0]).astype(bool)
        self.w_kc2en[:, 0][learning_rule] = np.maximum(self.w_kc2en[:, 0][learning_rule] - self.learning_rate, 0)

    def mask_pn2kc_weights_randomly(self):
        mask = np.zeros((self.nb_pn,self.nb_kc))
        for i in range(self.nb_kc):
            nr_dims = random.randrange(8,13)
            kept_dims = np.random.choice(self.nb_pn,nr_dims)
            mask[kept_dims,i] = 1
        self.w_pn2kc = self.w_pn2kc * mask

    def mask_pn2kc_weights_pca(self):
        mask = np.zeros((self.nb_pn,self.nb_kc))
        pn_as_dims = np.transpose(self.w_pn2kc)

        pca_instance = sklearn.decomposition.PCA(n_components=13)
        pca_instance.fit(pn_as_dims)

        dimprobs_unnorm = abs(np.matmul( pca_instance.explained_variance_, pca_instance.components_))
        dimprobs =  dimprobs_unnorm / dimprobs_unnorm.sum()

        for i in range(self.nb_kc):
            nr_dims = random.randrange(8,13)
            kept_dims = np.random.choice(self.nb_pn,nr_dims,p=dimprobs)
            mask[kept_dims,i] = 1
        self.w_pn2kc = self.w_pn2kc * mask

def generate_random_pn2kc_weights(nb_pn, nb_kc, dtype = np.float32):

    """
    Generate a random initial set of weights producing a fully-connected PN-KC network,
    with values in the interval [0,1), to be later modified during training.

    """
    np.random.seed(2019)
    w_pn2kc = np.random.rand(nb_pn,nb_kc).astype(dtype)
    normal_w_pn2kc = 42* w_pn2kc / np.linalg.norm(w_pn2kc,axis=0)
    return normal_w_pn2kc


def generate_pn2kc_weights(nb_pn, nb_kc, min_pn=8, max_pn=12, aff_pn2kc=None, nb_trials=100000, baseline=25000,
                           rnd=RNG, dtype=np.float32):
    """
    Create the synaptic weights among the Projection Neurons (PNs) and the Kenyon Cells (KCs).
    Choose the first sample that has dispersion below the baseline (early stopping), or the
    one with the lower dispersion (in case non of the samples' dispersion is less than the
    baseline).

    :param nb_pn:       the number of the Projection Neurons (PNs)
    :param nb_kc:       the number of the Kenyon Cells (KCs)
    :param min_pn:
    :param max_pn:
    :param aff_pn2kc:   the number of the PNs connected to every KC (usually 28-34)
                        if the number is less than or equal to zero it creates random values
                        for each KC in range [28, 34]
    :param nb_trials:   the number of trials in order to find a acceptable sample
    :param baseline:    distance between max-min number of projections per PN
    :param rnd:
    :type rnd: np.random.RandomState
    :param dtype:
    """

    dispersion = np.zeros(nb_trials)
    best_pn2kc = None

    for trial in range(nb_trials):
        pn2kc = np.zeros((nb_pn, nb_kc), dtype=dtype)

        if aff_pn2kc is None or aff_pn2kc <= 0:
            vaff_pn2kc = rnd.randint(min_pn, max_pn + 1, size=nb_pn)
        else:
            vaff_pn2kc = np.ones(nb_pn) * aff_pn2kc

        # go through every kenyon cell and select a nb_pn PNs to make them afferent
        for i in range(nb_pn):
            pn_selector = rnd.permutation(nb_kc)
            pn2kc[i, pn_selector[:vaff_pn2kc[i]]] = 1

        # This selections mechanism can be used to restrict the distribution of random connections
        #  compute the sum of the elements in each row giving the number of KCs each PN projects to.
        pn2kc_sum = pn2kc.sum(axis=0)
        dispersion[trial] = pn2kc_sum.max() - pn2kc_sum.min()
        # pn_mean = pn2kc_sum.mean()

        # Check if the number of projections per PN is balanced (min max less than baseline)
        #  if the dispersion is below the baseline accept the sample
        if dispersion[trial] <= baseline: return pn2kc

        # cache the pn2kc with the least dispersion
        if best_pn2kc is None or dispersion[trial] < dispersion[:trial].min():
            best_pn2kc = pn2kc

    # if non of the samples have dispersion lower than the baseline,
    # return the less dispersed one
    return best_pn2kc


if __name__ == "__main__":
    from world import load_world, load_routes
    from agent.visualiser import Visualiser
    from world.conditions import Hybrid

    world = load_world()
    routes = load_routes()
    routes[0].condition = Hybrid(tau_x=.1, tau_phi=np.pi)
    world.add_route(routes[0])

    nn = WillshawNet(nb_channels=3)
    nn.update = True
    vis = Visualiser(mode="panorama")
    vis.reset()

    x, y, z = np.zeros(3)
    phi = 0.

    def world_snapshot(width=None, height=None):
        global x, y, z, phi
        return world.draw_panoramic_view(x, y, z, phi, update_sky=False, include_ground=.3, include_sky=1.,
                                         width=width, length=width, height=height)

    for x, y, z, phi in world.routes[-1]:

        if vis.is_quit():
            print ("QUIT!")
            break

        img = world.draw_panoramic_view(x, y, z, phi)
        inp = np.array(img).reshape((-1, 3))
        en = nn(inp.flatten())

        vis.update_main(world_snapshot, caption="PN: %3d, KC: %3d, EN: %3d" % (nn.pn.sum(), nn.kc.sum(), en.sum()))
