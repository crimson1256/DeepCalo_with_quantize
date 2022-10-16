import tensorflow.keras as ks
import tensorflow.keras.backend as K

"""
    Slices a tensor on 1-D dimension from start to end.
    Example : to slice tensor x[5:10], set Slice_tensor1D(5, 10)(x)
    
"""
    

class Slice_tensor1D(ks.layers.Layer):
    
    def __init__(self, start=None, end=None,**kwargs):
        super().__init__(**kwargs)
        
       
        self.start = start
        self.end = end
        
       
    def call(self, inputs):
        assert(len(inputs.shape) == 2)
        assert(self.end > self.start)             
        return inputs[:, self.start:self.end]
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'start': self.start,
            'end': self.end,       
        })
        return config
        

