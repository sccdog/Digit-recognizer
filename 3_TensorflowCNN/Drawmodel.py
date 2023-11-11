import keras
from keras.models import load_model
from keras.utils.vis_utils import plot_model


model = load_model('models/model_TensorflowCNN5.h5')
keras.utils.plot_model(
        model,
        to_file='model.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
        dpi=900
    )

