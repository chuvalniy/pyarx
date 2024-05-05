# cls, x, y, w, h

# if y_pred center is in i'th cell that i'th cell is responsible for the prediction

def identity(y_true, y_pred):
    """
    Identity function for a bounding box loss function.
    Set 1 if cell is responsible for a bounding box prediction else 0.

    :param y_true: Ground truth values.
    :param y_pred: Predicted values.
    :return: Tensor of shape y_true.
    """
    pass


