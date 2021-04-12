"""
构造bm25的向量化方法
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.utils.validation import check_is_fitted
import scipy.sparse as sp
from sklearn.preprocessing import normalize


class BM25Vectorizer(CountVectorizer):

        def __init__(self, k=1.2, b=0.75, input='content', encoding='utf-8',
                     decode_error='strict', strip_accents=None, lowercase=True,
                     preprocessor=None, tokenizer=None, analyzer='word',
                     stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                     ngram_range=(1, 1), max_df=1.0, min_df=1,
                     max_features=None, vocabulary=None, binary=False,
                     dtype=np.int64, norm='l2', use_idf=True, smooth_idf=True,
                     sublinear_tf=False):
            super(BM25Vectorizer, self).__init__(
                input=input, encoding=encoding, decode_error=decode_error,
                strip_accents=strip_accents, lowercase=lowercase,
                preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
                stop_words=stop_words, token_pattern=token_pattern,
                ngram_range=ngram_range, max_df=max_df, min_df=min_df,
                max_features=max_features, vocabulary=vocabulary, binary=binary,
                dtype=dtype)

            self._tfidf = BM25Transformer(k=k, b=b, norm=norm, use_idf=use_idf,
                                           smooth_idf=smooth_idf,
                                           sublinear_tf=sublinear_tf)

        # Broadcast the TF-IDF parameters to the underlying transformer instance
        # for easy grid search and repr

        @property
        def norm(self):
            return self._tfidf.norm

        @norm.setter
        def norm(self, value):
            self._tfidf.norm = value

        @property
        def use_idf(self):
            return self._tfidf.use_idf

        @use_idf.setter
        def use_idf(self, value):
            self._tfidf.use_idf = value

        @property
        def smooth_idf(self):
            return self._tfidf.smooth_idf

        @smooth_idf.setter
        def smooth_idf(self, value):
            self._tfidf.smooth_idf = value

        @property
        def sublinear_tf(self):
            return self._tfidf.sublinear_tf

        @sublinear_tf.setter
        def sublinear_tf(self, value):
            self._tfidf.sublinear_tf = value

        @property
        def idf_(self):
            return self._tfidf.idf_

        def fit(self, raw_documents, y=None):
            """Learn vocabulary and idf from training set.

            Parameters
            ----------
            raw_documents : iterable
                an iterable which yields either str, unicode or file objects

            Returns
            -------
            self : BM25Vectorizer
            """
            X = super(BM25Vectorizer, self).fit_transform(raw_documents)
            self._tfidf.fit(X)
            return self

        def fit_transform(self, raw_documents, y=None):
            """Learn vocabulary and idf, return term-document matrix.

            This is equivalent to fit followed by transform, but more efficiently
            implemented.

            Parameters
            ----------
            raw_documents : iterable
                an iterable which yields either str, unicode or file objects

            Returns
            -------
            X : sparse matrix, [n_samples, n_features]
                Tf-idf-weighted document-term matrix.
            """
            # raw_documents = np.array(raw_documents).astype(np.float64)
            X = super(BM25Vectorizer, self).fit_transform(raw_documents)
            self._tfidf.fit(X)
            # X is already a transformed view of raw_documents so
            # we set copy to False
            return self._tfidf.transform(X, copy=False)

        def transform(self, raw_documents, copy=True):
            """Transform documents to document-term matrix.

            Uses the vocabulary and document frequencies (df) learned by fit (or
            fit_transform).

            Parameters
            ----------
            raw_documents : iterable
                an iterable which yields either str, unicode or file objects

            copy : boolean, default True
                Whether to copy X and operate on the copy or perform in-place
                operations.

            Returns
            -------
            X : sparse matrix, [n_samples, n_features]
                Tf-idf-weighted document-term matrix.
            """
            check_is_fitted(self, '_tfidf', 'The tfidf vector is not fitted')

            X = super(BM25Vectorizer, self).transform(raw_documents)
            return self._tfidf.transform(X, copy=False)


class BM25Transformer(BaseEstimator, TransformerMixin):

    def __init__(self, k=1.2, b=0.75, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        self.k = k
        self.b = b
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    def fit(self, X, y=None):
        """Learn the idf vector (global term weights)

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        """
        # print(X.shape)
        # print(type(X))
        # print(X.dtype)
        # X = X.astype(np.int8)
        print(X)
        self.avdl = np.mean(np.sum(X.toarray(), axis=-1))  # 平均长度

        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)

            # perform idf smoothing if required
            df += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            # log+1 instead of log makes sure terms with zero idf don't get
            # suppressed entirely.
            idf = np.log(float(n_samples) / df) + 1.0
            self._idf_diag = sp.spdiags(idf, diags=0, m=n_features,
                                        n=n_features, format='csr')

        return self

    def transform(self, X, copy=True):
        """Transform a count matrix to a tf or tf-idf representation

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts

        copy : boolean, default True
            Whether to copy X and operate on the copy or perform in-place
            operations.

        Returns
        -------
        vectors : sparse matrix, [n_samples, n_features]
        """
        # 计算中间项
        # X = X.astype(np.int8)
        d = np.sum(X.toarray(), axis=-1).reshape((X.toarray().shape[0], 1))  # 一句话的总词数
        # d = d.astype(np.float8)
        tf = X.toarray()/d
        # tf = tf.astype(np.float8)
        p = 1 - self.b + self.b * (d/self.avdl)
        mid_part = (self.k + 1)*tf/(tf + self.k*p)

        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.floating):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

        if self.use_idf:
            check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            # *= doesn't work
            X = X * self._idf_diag

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        X = X.toarray() * mid_part
        return X

    @property
    def idf_(self):
        # if _idf_diag is not set, this will raise an attribute error,
        # which means hasattr(self, "idf_") is False
        return np.ravel(self._idf_diag.sum(axis=0))


def _document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(sp.csc_matrix(X, copy=False).indptr)
