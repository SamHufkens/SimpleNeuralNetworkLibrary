class Loss:

  @staticmethod
  def binary_crossentropy(a2, y_pred):
    m = y_pred.shape[1]
    epsilon = 1e-5
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = - (1 / m) * np.sum(a2 * np.log(y_pred) + (1 - a2) * np.log(1 - y_pred))
    return loss

  @staticmethod
  def categorical_crossentropy(y, y_pred):
        return -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=0))