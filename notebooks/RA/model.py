import tensorflow as tf
from typing import List, Optional, Tuple, Union, Dict

class DNNModel:
    _DEFAULT_OPTIMIZER = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    _DEFAULT_LOSS = tf.keras.losses.MeanSquaredError()
    _DEFAULT_METRICS = [tf.keras.metrics.MeanSquaredError()]

    def __init__(
        self, 
        input_dim: Optional[int] = None, 
        hidden_layers: Optional[List[Dict]] = None,
        output_dim: Optional[int] = 1
    ) -> None:
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.model = None
    
    def build(
        self, 
        loss: tf.keras.losses.Loss = None, 
        metrics: Optional[List] = None, 
        optimizer: tf.keras.optimizers.Optimizer = None
    ) -> None:
        ins = tf.keras.Input(shape=(self.input_dim,))
        x = ins
        for h in self.hidden_layers:
            if 'activation' in h:
                x = tf.keras.layers.Dense(h['dim'], activation=h['activation'])(x)
            else:
                x = tf.keras.layers.Dense(h['dim'])(x)
        outs = tf.keras.layers.Dense(self.output_dim)(x)  # Linear output layer 
        self.model = tf.keras.Model(inputs=ins, outputs=outs)
        print(self.model.summary())
        
        # model compile
        if loss is None:
            loss = self._DEFAULT_LOSS
        if metrics is None:
            metrics = self._DEFAULT_METRICS
        if optimizer is None:
            optimizer = self._DEFAULT_OPTIMIZER
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def fit(self, x, y, **kwargs):
        self.model.fit(x, y, validation_split=0.2, **kwargs)
        
    def predict(self, x):
        return self.model.predict(x)
        
        