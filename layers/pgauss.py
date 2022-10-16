import tensorflow.keras as ks
import tensorflow.keras.backend as K

class ParametricGauss(ks.layers.Layer):

    def __init__(self, flipped=False, **kwargs):
        self.flipped = flipped
        super(ParametricGauss, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create trainable weight variables for this layer.
        self.alpha = self.add_weight(name='alpha',
                                      shape=(1,),
                                      initializer='uniform',
                                      trainable=True)

        super(ParametricGauss, self).build(input_shape)

    def call(self, x):
        def pgauss(x, sigma, flipped=False):
            if flipped:
                return 1 - K.exp(-(x/sigma)**2)
            else:
                return K.exp(-(x/sigma)**2)
        return pgauss(x, self.alpha, flipped=self.flipped)

        # def parametric_mrelu(x, alpha_neg, alpha_pos, flipped=False):
        #
        #     alpha_neg = K.abs(alpha_neg)
        #     alpha_pos = K.abs(alpha_pos)
        #
        #     if flipped:
        #         return 1 - K.minimum(K.maximum(1 - x*alpha_pos, 0), K.maximum(1 + x*alpha_neg, 0))
        #     else:
        #         return K.minimum(K.maximum(1 - x*alpha_pos, 0), K.maximum(1 + x*alpha_neg, 0))
        #
        # def parametric_gate(x, beta_1, beta_2, flipped=False):
        #
        #     # alpha = K.abs(alpha)
        #     beta_1 = K.abs(beta_1)
        #     beta_2 = K.abs(beta_2)
        #
        #     def gauss(x, beta):
        #         return K.exp(-beta * x**2)
        #
        #     # sigmoid = 1. / (1. + K.exp(alpha * K.abs(x)))
        #     gaussception = (gauss(1. - gauss(x, beta_2), beta_1) - K.exp(-beta_1)) / (1. - K.exp(-beta_1))
        #
        #     # return 2. * sigmoid * gaussception
        #     return gaussception
        #
        # return parametric_gate(x, self.beta_1, self.beta_2, flipped=True)
        # return parametric_mrelu(x, self.beta_1, self.beta_2, flipped=True)

    def compute_output_shape(self, input_shape):
        return input_shape
