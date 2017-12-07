import cv2
import numpy as np
import matplotlib.pyplot as plt
# import ipdb
import scipy
import chumpy as ch
from chumpy.ch import MatVecMult, Ch, depends_on


def pixelLayerPriors(masks):

    return np.sum(masks, axis=2) / masks.shape[-1]

def globalLayerPrior(masks):

    return np.sum(masks) / masks.size

def modelLogLikelihoodRobust(image, template, testMask, backgroundModel, layerPriors, variances):
    likelihood = pixelLikelihoodRobust(image, template, testMask, backgroundModel,  layerPriors, variances)
    liksum = np.sum(np.log(likelihood))


    return liksum

def modelLogLikelihoodRobustCh(image, template, testMask, backgroundModel, layerPriors, variances):
    likelihood = pixelLikelihoodRobustCh(image, template, testMask, backgroundModel,  layerPriors, variances)
    liksum = ch.sum(ch.log(likelihood))

    return liksum


def modelLogLikelihood(image, template, testMask, backgroundModel, variances):
    likelihood = pixelLikelihood(image, template, testMask, backgroundModel, variances)
    liksum = np.sum(np.log(likelihood))

def modelLogLikelihoodCh(image, template, testMask, backgroundModel, variances):
    logLikelihood = logPixelLikelihoodCh(image, template, testMask, backgroundModel, variances)

    return ch.sum(logLikelihood)

def pixelLikelihoodRobust(image, template, testMask, backgroundModel, layerPrior, variances):
    sigma = np.sqrt(variances)
    mask = testMask
    if backgroundModel == 'FULL':
        mask = np.ones(image.shape[0:2])
    # mask = np.repeat(mask[..., np.newaxis], 3, 2)
    repPriors = np.tile(layerPrior, image.shape[0:2])
    # sum = np.sum(np.log(layerPrior * scipy.stats.norm.pdf(image, location = template, scale=np.sqrt(variances) ) + (1 - repPriors)))
    # uniformProbs = np.ones(image.shape)

    foregroundProbs = np.prod(1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (image - template)**2 / (2 * variances)) * layerPrior, axis=2) + (1 - repPriors)
    return foregroundProbs * mask + (1-mask)

def pixelLikelihoodRobustSQErrorCh(sqeRenderer, testMask, backgroundModel, layerPrior, variances):
    sigma = ch.sqrt(variances)
    mask = testMask
    if backgroundModel == 'FULL':
        mask = np.ones(sqeRenderer.r.shape[0:2])
    # mask = np.repeat(mask[..., np.newaxis], 3, 2)
    repPriors = ch.tile(layerPrior, sqeRenderer.r.shape[0:2])
    # sum = np.sum(np.log(layerPrior * scipy.stats.norm.pdf(image, location = template, scale=np.sqrt(variances) ) + (1 - repPriors)))
    # uniformProbs = np.ones(image.shape)

    probs = ch.exp( - (sqeRenderer) / (2 * variances)) * (1./(sigma * np.sqrt(2 * np.pi)))
    foregroundProbs = (probs[:,:,0] * probs[:,:,1] * probs[:,:,2]) * layerPrior + (1 - repPriors)

    return foregroundProbs * mask + (1-mask)



def pixelLikelihoodRobustCh(image, template, testMask, backgroundModel, layerPrior, variances):
    sigma = ch.sqrt(variances)
    mask = testMask
    if backgroundModel == 'FULL':
        mask = np.ones(image.shape[0:2])
    # mask = np.repeat(mask[..., np.newaxis], 3, 2)
    repPriors = ch.tile(layerPrior, image.shape[0:2])
    # sum = np.sum(np.log(layerPrior * scipy.stats.norm.pdf(image, location = template, scale=np.sqrt(variances) ) + (1 - repPriors)))
    # uniformProbs = np.ones(image.shape)

    probs = ch.exp( - (image - template)**2 / (2 * variances)) * (1./(sigma * np.sqrt(2 * np.pi)))
    foregroundProbs = (probs[:,:,0] * probs[:,:,1] * probs[:,:,2]) * layerPrior + (1 - repPriors)
    return foregroundProbs * mask + (1-mask)


import chumpy as ch
from chumpy import depends_on, Ch

class NLLRobustModel(Ch):
    terms = ['useMask']
    terms = ['Q', 'variances']
    dterms = ['renderer', 'groundtruth']

    def compute_r(self):
        return -np.sum(np.log(self.prob))

    def compute_dr_wrt(self, wrt):
        if wrt is self.renderer:
            # fgMask = np.array(self.renderer.image_mesh_bool([0])).astype(np.bool)
            # visibility = self.renderer.visibility_image
            # visible = visibility != 4294967295
            visible = self.renderer.indices_image!=0
            fgMask = visible

            dr = (-1./(self.prob) * fgMask * self.fgProb[:,:,0]*self.fgProb[:,:,1]*self.fgProb[:,:,2] * self.Q[:, :])[:, :, None] * ((self.groundtruth.r - self.renderer.r)/self.variances.r)


            return dr.ravel()

    @depends_on(dterms)
    def fgProb(self):
        return np.exp(- (self.renderer.r - self.groundtruth.r) ** 2 / (2 * self.variances.r)) * (1. / (np.sqrt(self.variances.r) * np.sqrt(2 * np.pi)))

    @depends_on(dterms)
    def prob(self):
        h = self.renderer.r.shape[0]
        w = self.renderer.r.shape[1]

        occProb = np.ones([h, w])
        bgProb = np.ones([h, w])

        # visibility = self.renderer.visibility_image
        # visible = visibility != 4294967295
        try:
            self.useMask
        except:
            self.useMask = False

        if self.useMask:
            visible = self.renderer.indices_image != 0
            fgMask = visible
        else:
            fgMask = np.ones_like(self.renderer.indices_image.astype(np.bool))


        # fgMask = np.array(self.renderer.image_mesh_bool([0])).astype(np.bool)

        errorFun = fgMask[:, :]*(self.Q[:, :] * self.fgProb[:,:,0]*self.fgProb[:,:,1]*self.fgProb[:,:,2] + (1-self.Q[:, :]))+ (1- fgMask[:, :])

        return errorFun

    # @depends_on(dterms)
    # def prob(self):
    #     h = self.renderer.r.shape[0]
    #     w = self.renderer.r.shape[1]
    #
    #     occProb = np.ones([h, w])
    #     bgProb = np.ones([h, w])
    #
    #     fgMask = np.array(self.renderer.image_mesh_bool([0])).astype(np.bool)
    #
    #     errorFun = fgMask[:, :, None] * ((self.Q[0][:, :, None] * self.fgProb) + (self.Q[1] * occProb + self.Q[2] * bgProb)[:, :, None]) + (1 - fgMask[:, :, None])
    #
    #     return errorFun

class LogRobustModel(Ch):
    terms = ['useMask']
    dterms = ['renderer', 'groundtruth', 'foregroundPrior', 'variances']

    def compute_r(self):
        return self.logProb()

    def compute_dr_wrt(self, wrt):
        if wrt is self.renderer:
            return self.logProb().dr_wrt(self.renderer)

    def logProb(self):
        # visibility = self.renderer.visibility_image
        # visible = visibility != 4294967295
        try:
            self.useMask
        except:
            self.useMask = False

        if self.useMask:
            visible = self.renderer.indices_image != 0
        else:
            visible = np.ones_like(self.renderer.indices_image.astype(np.bool))

        # visible = np.array(self.renderer.image_mesh_bool([0])).copy().astype(np.bool)

        return ch.log(pixelLikelihoodRobustCh(self.groundtruth, self.renderer, visible, 'MASK', self.foregroundPrior, self.variances))

class LogGaussianModel(Ch):
    terms = ['useMask']
    dterms = ['renderer', 'groundtruth', 'variances']

    def compute_r(self):
        return self.logProb()

    def compute_dr_wrt(self, wrt):
        if wrt is self.renderer:
            return self.logProb().dr_wrt(self.renderer)

    def logProb(self):
        # visibility = self.renderer.visibility_image
        # visible = visibility != 4294967295
        try:
            self.useMask
        except:
            self.useMask = False

        if self.useMask:
            visible = self.renderer.indices_image != 0 # assumes the first mesh is the background cube.
        else:
            visible = np.ones_like(self.renderer.indices_image.astype(np.bool))

        # visible = np.array(self.renderer.image_mesh_bool([0])).copy().astype(np.bool)

        return logPixelLikelihoodCh(self.groundtruth, self.renderer, visible, 'MASK', self.variances)

def pixelLikelihood(image, template, testMask, backgroundModel, variances):
    sigma = np.sqrt(variances)
    # sum = np.sum(np.log(layerPrior * scipy.stats.norm.pdf(image, location = template, scale=np.sqrt(variances) ) + (1 - repPriors)))
    mask = testMask
    if backgroundModel == 'FULL':
        mask = np.ones(image.shape[0:2])
    # mask = np.repeat(mask[..., np.newaxis], 3, 2)
    uniformProbs = np.ones(image.shape[0:2])
    normalProbs = np.prod((1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (image - template)**2 / (2 * variances))),axis=2)
    return normalProbs * mask + (1-mask)

def logPixelLikelihoodCh(image, template, testMask, backgroundModel, variances):
    sigma = ch.sqrt(variances)
    # sum = np.sum(np.log(layerPrior * scipy.stats.norm.pdf(image, location = template, scale=np.sqrt(variances) ) + (1 - repPriors)))
    mask = testMask
    if backgroundModel == 'FULL':
        mask = np.ones(image.shape[0:2])
    # mask = np.repeat(mask[..., np.newaxis], 3, 2)
    uniformProbs = np.ones(image.shape[0:2])
    logprobs =   (-(image - template)**2 / (2. * variances))  - ch.log((sigma * np.sqrt(2.0 * np.pi)))
    pixelLogProbs = logprobs[:,:,0] + logprobs[:,:,1] + logprobs[:,:,2]
    return pixelLogProbs * mask

def logPixelLikelihoodErrorCh(sqerrors, testMask, backgroundModel, variances):
    sigma = ch.sqrt(variances)
    # sum = np.sum(np.log(layerPrior * scipy.stats.norm.pdf(image, location = template, scale=np.sqrt(variances) ) + (1 - repPriors)))
    mask = testMask
    if backgroundModel == 'FULL':
        mask = np.ones(sqerrors.shape[0:2])
    # mask = np.repeat(mask[..., np.newaxis], 3, 2)
    uniformProbs = np.ones(sqerrors.shape[0:2])
    logprobs =   (-(sqerrors) / (2. * variances))  - ch.log((sigma * np.sqrt(2.0 * np.pi)))
    pixelLogProbs = logprobs[:,:,0] + logprobs[:,:,1] + logprobs[:,:,2]
    return pixelLogProbs * mask

def pixelLikelihoodCh(image, template, testMask, backgroundModel, layerPrior, variances):
    sigma = ch.sqrt(variances)
    mask = testMask
    if backgroundModel == 'FULL':
        mask = np.ones(image.shape[0:2])
    # mask = np.repeat(mask[..., np.newaxis], 3, 2)
    repPriors = ch.tile(layerPrior, image.shape[0:2])
    # sum = np.sum(np.log(layerPrior * scipy.stats.norm.pdf(image, location = template, scale=np.sqrt(variances) ) + (1 - repPriors)))
    # uniformProbs = np.ones(image.shape)

    probs = ch.exp( - (image - template)**2 / (2 * variances)) * (1./(sigma * np.sqrt(2 * np.pi)))
    foregroundProbs = (probs[:,:,0] * probs[:,:,1] * probs[:,:,2])
    return foregroundProbs * mask + (1-mask)

def layerPosteriorsRobust(image, template, testMask, backgroundModel, layerPrior, variances):

    sigma = np.sqrt(variances)
    mask = testMask
    if backgroundModel == 'FULL':
        mask = np.ones(image.shape[0:2])
    # mask = np.repeat(mask[..., np.newaxis], 3, 2)
    repPriors = np.tile(layerPrior, image.shape[0:2])
    foregroundProbs = np.prod(1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (image - template)**2 / (2 * variances)) * layerPrior, axis=2)
    backgroundProbs = np.ones(image.shape)
    outlierProbs = (1-repPriors)
    lik = pixelLikelihoodRobust(image, template, testMask, backgroundModel, layerPrior, variances)
    # prodlik = np.prod(lik, axis=2)
    # return np.prod(foregroundProbs*mask, axis=2)/prodlik, np.prod(outlierProbs*mask, axis=2)/prodlik

    return foregroundProbs*mask/lik, outlierProbs*mask/lik

def layerPosteriorsRobustCh(image, template, testMask, backgroundModel, layerPrior, variances):

    sigma = ch.sqrt(variances)
    mask = testMask
    if backgroundModel == 'FULL':
        mask = np.ones(image.shape[0:2])
    # mask = np.repeat(mask[..., np.newaxis], 3, 2)
    repPriors = ch.tile(layerPrior, image.shape[0:2])
    probs = ch.exp( - (image - template)**2 / (2 * variances))  * (1/(sigma * np.sqrt(2 * np.pi)))
    foregroundProbs =  probs[:,:,0] * probs[:,:,1] * probs[:,:,2] * layerPrior
    backgroundProbs = np.ones(image.shape)
    outlierProbs = ch.Ch(1-repPriors)
    lik = pixelLikelihoodRobustCh(image, template, testMask, backgroundModel, layerPrior, variances)
    # prodlik = np.prod(lik, axis=2)
    # return np.prod(foregroundProbs*mask, axis=2)/prodlik, np.prod(outlierProbs*mask, axis=2)/prodlik

    return foregroundProbs*mask/lik, outlierProbs*mask/lik

