#!/usr/bin/env python3

import plaidml.keras
import os
plaidml.keras.install_backend()
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


from data_processing import t_sne_data_processing
from MulticoreTSNE import MulticoreTSNE as TSNE
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from constants import CONFIG
import multiprocessing
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_sklearn_tsne(perplexity):
  x, _, _, y, _, _ = t_sne_data_processing()
  # _, x, _, _, y, _ = t_sne_data_processing()
  # _, _, x, _, _, y = t_sne_data_processing()
  print("cpu_count:", multiprocessing.cpu_count())
  tsne = TSNE(
      n_jobs=multiprocessing.cpu_count() // 2,
      n_iter=1000,
      n_components=2,
      verbose=1,
      # perplexity=perplexity,
  )
  proj = tsne.fit_transform(x)
  cmp = plt.get_cmap("jet", len(y))
  fig = plt.figure(figsize=(10, 10))
  # ax = Axes3D(fig)
  # ax.set_xlabel("x")
  # ax.set_ylabel("y")
  # ax.set_zlabel("z")
  for i in range(len(y)):
    select_flag = y == y[i]
    plt_latent = proj[select_flag, :]
    plt.scatter(
      plt_latent[:, 0],
      plt_latent[:, 1],
      color=cmp(i),
      marker=f"${i}$",
      s=100
    )
    # ax.scatter(
    #   plt_latent[:, 0],
    #   plt_latent[:, 1],
    #   plt_latent[:, 2],
    #   color=cmp(i),
    #   marker=f"${i}$"
    # )
  # plt.title("t-SNE")
  arr = np.array(
      [
          [
              p[0],
              p[1],
              i,
          ]
          for p, i in zip(proj, list(range(len(y))))
      ]
  )
  arr = arr[np.argsort(arr[:, 1])[::-1]]
  for i in range(11):
    arr[(i * 11):((i + 1) * 11)] = arr[np.array([i * 11] * 11) + np.argsort(arr[(i * 11):((i + 1) * 11), 0])]
  arr = np.array(arr)[:, 2]
  arr = arr.reshape(11, 11)
  # print("perplexity = " + str(perplexity))
  for a in arr:
    a = [int(aa) + 1 for aa in a]
    print(*a, sep=", ")
  # plt.show()
  # plt.savefig("./temp/" + "perplexity_" + str(perplexity) + ".jpg")


if __name__ == "__main__":
  for i in range(1):
    perplexity = (i + 1) * 100.0
    plot_sklearn_tsne(perplexity)
