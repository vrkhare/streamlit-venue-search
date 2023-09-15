from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class BM25Vectorizer(TfidfVectorizer):

  def __init__(self, b=0.75, k1=1.2, **kwargs):
        self.b = b 
        self.k1 = k1
        super().__init__(norm=None, **kwargs)

  # Override to calculate avgdl
  def fit(self, X, y=None):
    super().fit(X)
    self._avdl = self.idf_.sum() / self.idf_.shape[0]
    return self
    
  # Override to apply BM25 normalization
  def transform(self, X):
    X = super().transform(X)
    
    # Densify matrix to vector
    X = X.toarray() 
    
    d = X.sum(1)

    # Apply BM25 normalization
    avdl_vec = np.full_like(d, self._avdl)
    term_weights = self.k1 * (1 - self.b + self.b * avdl_vec / d)[:, None]  
    X = X / (X + term_weights)
    
    return X