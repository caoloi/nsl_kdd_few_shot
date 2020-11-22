from keras import backend as K
from classifications import calc_centers


def ml_categorical_classification(x_support, y_support, model):
  def m_c_classification(y_true, y_pred):
    # centers = calc_centers(x_support, y_support, model)
    return K.variable(0.0)
  return m_c_classification
