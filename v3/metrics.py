import numpy as np

def metrics(
    model_pred_norm: np.ndarray,
    bilinear_pred_norm: np.ndarray,
    target_norm: np.ndarray,
    precip_mean: float,
    precip_std: float
) -> dict:
    """
    Compare performance of the model vs. a bilinear interpolation baseline over the entire test set.
    
    Computes the following metrics in both normalized and unnormalized precipitation space:
      - Mean Squared Error (MSE)
      - Mean Absolute Error (MAE)
      - Bias (Mean Error)
      - Pearson Correlation Coefficient (corr)
    
    Args:
        model_pred_norm (np.ndarray):
            Model predictions in normalized log-space, shape (N, fLat, fLon).
        bilinear_pred_norm (np.ndarray):
            Bilinear interpolation predictions in normalized log-space, shape (N, fLat, fLon).
            (This is basically the upsampled coarse input.)
        target_norm (np.ndarray):
            Ground truth data in normalized log-space, shape (N, fLat, fLon).
        precip_mean (float):
            The dataset-wide mean used in normalization (in log(1 + x) space).
        precip_std (float):
            The dataset-wide std used in normalization (in log(1 + x) space).
    
    Returns:
        dict:
            A nested dictionary of metrics for both "model" and "bilinear",
            each containing sub-keys for "mse", "mae", "bias", and "corr",
            and under each of those, sub-keys "normalized" and "unnormalized".
            
            For example:
            {
                "model": {
                    "mse":  {"normalized": ..., "unnormalized": ...},
                    "mae":  {"normalized": ..., "unnormalized": ...},
                    "bias": {"normalized": ..., "unnormalized": ...},
                    "corr": {"normalized": ..., "unnormalized": ...},
                },
                "bilinear": {
                    "mse":  {"normalized": ..., "unnormalized": ...},
                    ...
                }
            }
    """
    # ---------------------------------------------------------------------
    # Helper function: safe correlation to handle edge cases
    # ---------------------------------------------------------------------
    def _corrcoef(x, y):
        # If everything is constant or there's too little variation, corr can be NaN.
        # We'll return 0.0 in that scenario.
        if np.allclose(x, x[0]) or np.allclose(y, y[0]):
            return 0.0
        c = np.corrcoef(x, y)[0, 1]
        return np.nan_to_num(c, nan=0.0)

    # Flatten each array to 1D for metrics
    model_pred_norm_1d = model_pred_norm.ravel()
    bilinear_pred_norm_1d = bilinear_pred_norm.ravel()
    target_norm_1d = target_norm.ravel()

    # ---------------------------------------------------------------------
    # Normalized-space metrics
    # ---------------------------------------------------------------------
    def _mse(a, b):
        return np.mean((a - b) ** 2)
    def _mae(a, b):
        return np.mean(np.abs(a - b))
    def _bias(a, b):
        return np.mean(a - b)

    model_mse_norm = _mse(model_pred_norm_1d, target_norm_1d)
    model_mae_norm = _mae(model_pred_norm_1d, target_norm_1d)
    model_bias_norm = _bias(model_pred_norm_1d, target_norm_1d)
    model_corr_norm = _corrcoef(model_pred_norm_1d, target_norm_1d)

    bilinear_mse_norm = _mse(bilinear_pred_norm_1d, target_norm_1d)
    bilinear_mae_norm = _mae(bilinear_pred_norm_1d, target_norm_1d)
    bilinear_bias_norm = _bias(bilinear_pred_norm_1d, target_norm_1d)
    bilinear_corr_norm = _corrcoef(bilinear_pred_norm_1d, target_norm_1d)

    # ---------------------------------------------------------------------
    # Unnormalize predictions to mm space
    #   Inverse of normalized log-space:
    #       x_norm -> x_log = (x_norm * precip_std) + precip_mean
    #       x_mm = expm1( x_log ),   i.e. e^(x_log) - 1
    # ---------------------------------------------------------------------
    def _unnormalize(x_norm):
        x_log = (x_norm * precip_std) + precip_mean
        return np.expm1(x_log)

    model_pred_mm = _unnormalize(model_pred_norm_1d)
    bilinear_pred_mm = _unnormalize(bilinear_pred_norm_1d)
    target_mm = _unnormalize(target_norm_1d)

    # ---------------------------------------------------------------------
    # Unnormalized-space metrics
    # ---------------------------------------------------------------------
    model_mse_unnorm = _mse(model_pred_mm, target_mm)
    model_mae_unnorm = _mae(model_pred_mm, target_mm)
    model_bias_unnorm = _bias(model_pred_mm, target_mm)
    model_corr_unnorm = _corrcoef(model_pred_mm, target_mm)

    bilinear_mse_unnorm = _mse(bilinear_pred_mm, target_mm)
    bilinear_mae_unnorm = _mae(bilinear_pred_mm, target_mm)
    bilinear_bias_unnorm = _bias(bilinear_pred_mm, target_mm)
    bilinear_corr_unnorm = _corrcoef(bilinear_pred_mm, target_mm)

    # Structure the results
    out = {
        "model": {
            "mse": {
                "normalized": float(model_mse_norm),
                "unnormalized": float(model_mse_unnorm)
            },
            "mae": {
                "normalized": float(model_mae_norm),
                "unnormalized": float(model_mae_unnorm)
            },
            "bias": {
                "normalized": float(model_bias_norm),
                "unnormalized": float(model_bias_unnorm)
            },
            "corr": {
                "normalized": float(model_corr_norm),
                "unnormalized": float(model_corr_unnorm)
            }
        },
        "bilinear": {
            "mse": {
                "normalized": float(bilinear_mse_norm),
                "unnormalized": float(bilinear_mse_unnorm)
            },
            "mae": {
                "normalized": float(bilinear_mae_norm),
                "unnormalized": float(bilinear_mae_unnorm)
            },
            "bias": {
                "normalized": float(bilinear_bias_norm),
                "unnormalized": float(bilinear_bias_unnorm)
            },
            "corr": {
                "normalized": float(bilinear_corr_norm),
                "unnormalized": float(bilinear_corr_unnorm)
            }
        }
    }
    return out