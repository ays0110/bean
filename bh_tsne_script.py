import numpy as np
import bhtsne

data = np.load("../Data/numpy-data/e10-embeddings-squished.npy")
embedding_array = bhtsne.run_bh_tsne(data, initial_dims=data.shape[1], verbose=True)
np.save('../Data/numpy-data/e10-bh-results.npy',embedding_array) 
