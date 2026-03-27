import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	"""Mean Absolute Error between two numpy arrays.

	Args:
		y_true: Ground truth array.
		y_pred: Predicted array (same shape as y_true).

	Returns:
		Scalar MAE as float.
	"""
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)

	if y_true.shape != y_pred.shape:
		raise ValueError(f"Shapes must match: {y_true.shape} vs {y_pred.shape}")

	return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	"""Root Mean Squared Error between two numpy arrays.

	Args:
		y_true: Ground truth array.
		y_pred: Predicted array (same shape as y_true).

	Returns:
		Scalar RMSE as float.
	"""
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)

	if y_true.shape != y_pred.shape:
		raise ValueError(f"Shapes must match: {y_true.shape} vs {y_pred.shape}")

	return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

