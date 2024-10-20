import os
import numpy as np
import tensorflow as tf
import time
import argparse
import random
from sklearn.metrics import roc_auc_score, mean_absolute_error
import multiprocessing
import queue
import threading

from scipy.stats import rankdata, pearsonr, spearmanr, kendalltau

from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd


def _normalized_rmse(y_true, y_pred):
  return np.sqrt(mean_squared_error(y_true, y_pred)) / y_true.mean()

def cumulative_true(
    y_true,
    y_pred
) -> np.ndarray:
  """Calculates cumulative sum of lifetime values over predicted rank.
  Arguments:
    y_true: true lifetime values.
    y_pred: predicted lifetime values.
  Returns:
    res: cumulative sum of lifetime values over predicted rank.
  """
  df = pd.DataFrame({
      'y_true': y_true,
      'y_pred': y_pred,
  }).sort_values(
      by='y_pred', ascending=False)

  return (df['y_true'].cumsum() / df['y_true'].sum()).values

def gini_from_gain(df: pd.DataFrame) -> pd.DataFrame:
  """Calculates gini coefficient over gain charts.
  Arguments:
    df: Each column contains one gain chart. First column must be ground truth.
  Returns:
    gini_result: This dataframe has two columns containing raw and normalized
                 gini coefficient.
  """
  raw = df.apply(lambda x: 2 * x.sum() / df.shape[0] - 1.)
  normalized = raw / raw[0]
  return pd.DataFrame({
      'raw': raw,
      'normalized': normalized
  })[['raw', 'normalized']]
def corr_zero_inflate(labels, predicts):
    labels = np.array(labels)
    predicts = np.array(predicts)
    # labels = rankdata(np.array(labels))
    # predicts = rankdata(np.array(predicts))
    mask1 = labels > 0
    corr0 = spearmanr(labels, predicts)[0]
    corr1 = spearmanr(labels[mask1], predicts[mask1])[0]
    return corr0, corr1
