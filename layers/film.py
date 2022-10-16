import tensorflow.keras as ks
import tensorflow.keras.backend as K

class FiLM(ks.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        feature_map_shape, FiLM_tns_shape = input_shape
        self.height = feature_map_shape[1]
        self.width = feature_map_shape[2]
        self.n_feature_maps = feature_map_shape[-1]
        assert(int(2 * self.n_feature_maps)==FiLM_tns_shape[1])
        super().build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        conv_output, FiLM_tns = x

        # Duplicate in order to apply to entire feature maps
        # Taken from https://github.com/GuessWhatGame/neural_toolbox/blob/master/film_layer.py
        FiLM_tns = K.expand_dims(FiLM_tns, axis=[1])
        FiLM_tns = K.expand_dims(FiLM_tns, axis=[1])
        FiLM_tns = K.tile(FiLM_tns, [1, self.height, self.width, 1])

        # Split into gammas and betas
        gammas = FiLM_tns[:, :, :, :self.n_feature_maps]
        betas = FiLM_tns[:, :, :, self.n_feature_maps:]

        # Apply affine transformation
        return (1 + gammas) * conv_output + betas

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0]


class FiLM_active(ks.layers.Layer):

    def __init__(self, units=[64,64], activation='leakyrelu',
                 initialization='glorot_uniform', **kwargs):
        self.units = units
        self.activation = activation
        self.initialization = ks.initializers.get(initialization)
        super().__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        feature_map_shape, FiLM_vars_shape = input_shape
        self.n_feature_maps = feature_map_shape[-1]
        self.height = feature_map_shape[1]
        self.width = feature_map_shape[2]

        # Collect trainable weights
        trainable_weights = []

        # Create weights for hidden layers
        self.hidden_dense_layers = []
        for i,unit in enumerate(self.units):
            dense = ks.layers.Dense(unit,
                                    kernel_initializer=self.initialization)
            if i==0:
                build_shape = FiLM_vars_shape[:2]
            else:
                build_shape = (None,self.units[i-1])
            dense.build(build_shape)
            trainable_weights += dense.trainable_weights
            self.hidden_dense_layers.append(dense)

        # Create weights for output layer
        self.output_dense = ks.layers.Dense(2 * self.n_feature_maps, # assumes channel_last
                                            kernel_initializer=self.initialization)
        self.output_dense.build((None,self.units[-1]))
        trainable_weights += self.output_dense.trainable_weights

        # Pass on all collected trainable weights
        self._trainable_weights = trainable_weights

        super().build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        conv_output, FiLM_vars = x

        # Generate FiLM outputs
        tns = FiLM_vars
        for i in range(len(self.units)):
            tns = self.hidden_dense_layers[i](tns)
            tns = get_activation(activation=self.activation)(tns)
        FiLM_output = self.output_dense(tns)

        # Duplicate in order to apply to entire feature maps
        # Taken from https://github.com/GuessWhatGame/neural_toolbox/blob/master/film_layer.py
        FiLM_output = K.expand_dims(FiLM_output, axis=[1])
        FiLM_output = K.expand_dims(FiLM_output, axis=[1])
        FiLM_output = K.tile(FiLM_output, [1, self.height, self.width, 1])

        # Split into gammas and betas
        gammas = FiLM_output[:, :, :, :self.n_feature_maps]
        betas = FiLM_output[:, :, :, self.n_feature_maps:]

        # Apply affine transformation
        return (1 + gammas) * conv_output + betas

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0]
