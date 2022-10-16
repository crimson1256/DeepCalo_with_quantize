import tensorflow.keras as ks
import tensorflow.keras.backend as K

 
       
class Sum1D(ks.layers.Layer):

    #sum up a tensor on a axis
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

       
    def call(self, inputs):
        assert(len(inputs.shape) >=2 and len(inputs.shape)<=3)        
        return K.sum(inputs, axis=1)
    


class Mask_track(ks.layers.Layer):
    
    #fill an entire row with zeros if  the row contain a zero
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

      
    def call(self, inputs):
        assert(len(inputs.shape) == 3),'input shape with batch should be rank 3'      
        return K.cast(K.all(K.not_equal(inputs, 0), axis=2, keepdims=True), 'float32')
    






