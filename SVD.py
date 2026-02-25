"""
Singular Value Decomposition helper for whitening and alignment utilities.

Overview
--------
This module exposes a single public class, ``SingularValueDecomposition``,
which wraps ``numpy.linalg.svd`` and lazily computes a rich set of derived
quantities (pseudo-inverse, matrix square root, whitener, etc.).

It is used internally by ``Whitening.SpatialWhiteningDecomposition`` to:

- Decompose sensor covariance matrices for whitening (ZCA / PCA-style).
- Handle rank-deficient inputs via a configurable tolerance threshold and
  an optional hard ``maxRank`` ceiling.
- Provide stable matrix square roots and inverse square roots needed by CSP
  and SSA spatial filter computations.

All derived properties are computed on demand and cached so that repeated
access does not trigger redundant computation.

Public API
----------
``SingularValueDecomposition(A, maxRank=None, tol='auto', keepCopy=True, hermitian=False)``
    Decompose ``A`` and expose: ``U``, ``s``, ``V``, ``r`` (effective rank),
    ``pinv``, ``sqrtm``, ``isqrtm``, ``whitener``, ``reconstruction``, and more.
    See the class docstring for the full property list.

Notes
-----
- All properties follow the convention  ``A = U @ diag(s) @ V.H``  where
  ``U`` and ``V`` are orthonormal and ``s`` contains singular values in
  descending order.
- Setting ``hermitian=True`` delegates to the faster ``numpy.linalg.eigh``
  path internally; use this only when ``A`` is genuinely symmetric/Hermitian.
- The tolerance ``tol`` scales with ``max(A.shape) * eps(A.dtype)`` by default,
  which is consistent with MATLAB's ``rank`` and ``pinv`` conventions.
"""

__all__ = [
	'SingularValueDecomposition',
]

import numpy

DOC_LINES = []
def uncached_property( method ):
	prefix = '% 14s  ' % method.__name__
	lines = method.__doc__.rstrip().lstrip( '\n' ).split( '\n' )
	toRemove = len( lines[ 0 ] ) - len( lines[ 0 ].lstrip() ) if lines else 0
	for line in lines: DOC_LINES.append( prefix + line[ toRemove: ] ); prefix = ' ' * len( prefix )	
	return property( fget=method, doc=method.__doc__ )
	
def cached_property( method ):
	uncached_property( method ) # for the DOC_LINES side effect
	name = method.__name__
	def fget( self ):
		cached = self._cache.get( name, None )
		if cached is not None: return cached
		return self._cache.setdefault( name, method.__get__( self )() )
	fget.__name__ = method.__name__
	fget.__doc__ = method.__doc__
	return property( fget=fget, doc=method.__doc__ )

class SingularValueDecomposition( object ):
	"""
	d = SingularValueDecomposition(A)
	
	Manages the singular value decomposition of m-by-n matrix A into
	
	             d.U @ numpy.diag(d.s) @ d.V.H
	             
	(Note: for brevity, this class's docstrings use now-deprecated numpy.matrix
	notation: X.H as a shorthand for the Hermitian transpose X.conj().T, and
	X.I as a shorthand for the inverse, numpy.linalg.inv(X) ).
	
	U, s, V are computed on construction. The other properties are then cheap
	to compute, but to save space each one is only computed when requested (and
	cached at that time). The following are available:
	
	@PROPS@ - see documentation for the `.properties` property.
	"""
	def __init__( self, A, maxRank=None, tol='auto', keepCopy=True, hermitian=False ):
		"""
		Initialize with `hermitian=True` if you want to assert that `A` is equal to
		`A.conj().T`. If this is true, the underlying numpy algorithm can run faster
		(but if you say it is true when it is not, the result will be wrong).
		"""
		try:    ( self._U, self._s, self._Vh ) = numpy.linalg.svd( A, full_matrices=False, compute_uv=True, hermitian=hermitian )
		except: ( self._U, self._s, self._Vh ) = numpy.linalg.svd( A, full_matrices=False, compute_uv=True ) # backward compatibility with older numpy
		self._cache = {}
		self._original = A.copy() if keepCopy else None
		self._maxRank = maxRank
		self._defaultTolerance = max( A.shape ) * numpy.finfo( A.dtype ).eps
		self.tol = tol
		
	@uncached_property
	def A( self ):
		"""
		[m by n]  Original matrix (only stored if `keepCopy=True`
		          was passed to the constructor).
		"""
		return self._original
	original = A
	
	@uncached_property
	def m( self ):
		"[scalar]  The number of rows of the input matrix A."
		return self._U.shape[ 0 ]
	@uncached_property
	def n( self ):
		"[scalar]  The number of columns of the input matrix A."
		return self._Vh.shape[ 1 ]
	@uncached_property
	def tol( self ):
		"""
		[scalar]  The tolerance value (as a proportion of the
		          largest singular value) for estimating rank.
		"""
		return self._tol
	@tol.setter
	def tol( self, value ):
		self._tol = self._defaultTolerance if value in [ None, 'auto' ] else value
		self._cache.clear()
	@uncached_property
	def r( self ):
		"""
		[scalar]  Effective rank (estimated from A, or imposed
		          at a lower value).
		"""
		estimated = self.rank
		imposed = self._maxRank
		if imposed is not None and imposed < estimated: return max( 1, imposed )
		else: return estimated
	@r.setter
	def r( self, value ):
		self._maxRank = None if value == 'auto' else value
		self._cache.clear()
	imposedRank = r
		
	@cached_property
	def cond( self ):
		"""
		[scalar]  Condition number of A, estimated as the ratio
		          between the highest and lowest singular values.
		"""
		smin, smax = min( self._s ), max( self._s )
		return smax / smin if smin else numpy.inf
	@cached_property
	def rank( self ):
		"[scalar]  The rank as estimated from A, according to tol."
		return int( numpy.sum( self._s > self._tol * max( self._s ) ) )
	@cached_property
	def U( self ):
		"""
		[m by r]  Columns of U are an orthonormal basis for the
		          column space of A.
		"""
		return self._U[ :, :self.r ]
	@cached_property
	def s( self ):
		"[r]       Singular values of A, in decreasing order."
		return self._s[ : self.r ]
	@cached_property
	def S( self ):
		"""
		[r by r]  Diagonal matrix containing the singular values in
		          descreasing order.
		"""
		return numpy.diag( self.s )
	@cached_property
	def V( self ):
		"""
		[n by r]  Columns of V are an orthonormal basis for the row
		          space of A.
		"""
		return self._Vh[ :self.r, : ].conj().T
	@cached_property
	def leftNull( self ):
		"[m by min(m,n)-r]  The discarded columns of U."
		return self._U[ :, self.r: ]
	@cached_property
	def null( self ):
		"[n by min(m,n)-r]  The discarded columns of V."
		return self._Vh[ self.r:, : ].conj().T
	@cached_property
	def pinv( self ):
		"[n by m]  The pseudo-inverse of A."
		return numpy.dot( self.U * self.s[ None, : ] ** -1.0, self._Vh[ :self.r, : ] ).conj().T
	@cached_property
	def sqrtm( self ):
		"""
		[m by n]  X such that X @ X.H = (U @ S @ U.H)
		                  and X.H @ X = (V @ S @ V.H)
		          (hence if A is symmetric, X is the matrix square-
		          root of A).
		"""
		return numpy.dot( self.U * self.s[ None, : ] ** +0.5, self._Vh[ :self.r, : ] )
	@cached_property
	def isqrtm( self ):
		"""
		[n by m]  The (pseudo-)inverse of sqrtm, in other words, a
		          matrix X such that 
		          X.H @ X = (U @ S @ U.H).I (if invertible) and/or
		          X @ X.H = (V @ S @ V.H).I (if invertible).
		"""
		return numpy.dot( self.U * self.s[ None, : ] ** -0.5, self._Vh[ :self.r, : ] ).conj().T
	@cached_property
	def reconstruction( self ):
		"""
		[m by n]  Rank-r reconstruction of A, (made without using
		          the discarded columns of U and V.
		"""
		return numpy.dot( self.U * self.s[ None, : ]        , self._Vh[ :self.r, : ] )
	@cached_property
	def whitener( self ):
		"""
		[m by r]  P such that P.H @ A @ P is equal to eye(r). Only
		          valid when A is symmetric (or rather, Hermitian).
		"""
		return            self.U * self.s[ None, : ] ** -0.5
	
	def __repr__( self ):
		s = "<%s.%s instance at 0x%08X>" % ( self.__class__.__module__, self.__class__.__name__, id( self ) )
		s += "\n    U: % 3d by % 3d%s" % ( self.m, self.r, ' (of %d)' % self.rank if self.r < self.rank else '' )
		s += "\n    V: % 3d by % 3d%s" % ( self.n, self.r, ' (of %d)' % self.rank if self.r < self.rank else '' )
		return s
	

DOC_LINES += """
                  (Note: the above statements about A are only true when
                  d.rank == d.imposedRank; otherwise, they are true of
                  d.reconstruction.)
""".split( '\n' )
lines = SingularValueDecomposition.__doc__.split( '\n' )
for i, line in enumerate( lines ):
	if not line.strip().startswith( '@PROPS@' ): continue
	indent = line[ :len( line ) - len( line.lstrip() ) ]
	lines[ i ] = '\n'.join( indent + line for line in DOC_LINES )
	break
try: SingularValueDecomposition.__doc__ = '\n'.join( lines )
except: SingularValueDecomposition.properties = property( lambda self: None, doc='\n'.join( [ 'Properties of the SingularValueDecomposition class:', '' ] + DOC_LINES ) )
# (in Python 2.7, class __doc__ is not writable)
