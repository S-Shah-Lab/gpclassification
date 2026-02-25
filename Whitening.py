#!/usr/bin/env -s python3
"""
Spatial whitening, CSP/SSA filter computation, and covariance utilities.

Overview
--------
This module is the core linear-algebra layer for EEG spatial filtering in
this package.  It is used by the experiment runner
(``run_class_bci_competition_III_merged_spatial_filters_gpy.py``) to:

1. **Compute sensor covariance matrices** from multi-trial epoch arrays via
   ``Covariance``.
2. **Apply spatial filters** (CSP, SSA, or arbitrary matrices) to EEG
   epochs via ``ApplySpatialFilters``.
3. **Whiten sensor covariances** and derive optimal spatial filters
   through the ``SpatialWhiteningDecomposition`` class, which exposes:

   - ``Whiten``   — PCA/ZCA-style whitening of the sensor covariance.
   - ``Rayleigh`` — Rayleigh-quotient maximisation to obtain CSP-style
                   filters ordered by discriminability.
   - ``SSA``      — Stationary Subspace Analysis to identify components
                   whose statistics are stable across epochs.

Public API
----------
``ApplySpatialFilters(signal, spatialFilteringMatrix, sensorAxis=1)``
    Apply a spatial filter matrix to an array of signals using einsum.
    Handles 2-D ``(samples, channels)`` and 3-D ``(epochs, channels, samples)``
    inputs uniformly by referencing the sensor axis symbolically.

``Covariance(x, preservedAxis=-1)``
    Compute the unnormalised outer-product covariance along one axis of ``x``.
    Accepts raw NumPy arrays or MNE objects transparently.

``Check(x, against=None, name='')``
    Sanity-check helper: returns the maximum absolute deviation between ``x``
    and a reference matrix (default ``eye``).  Used for unit-test assertions
    and debugging whitening pipelines.

``SpatialWhiteningDecomposition(mixedSignals=None, sensorCovariance=None, ...)``
    Main class.  Accepts either raw epoch data or a pre-computed covariance
    matrix and provides whitening, CSP, and SSA decompositions.  All spatial
    filter matrices are stored as ``W`` (columns = filters) and ``A`` (columns =
    patterns / activations).

Dependencies
------------
- ``SVD.SingularValueDecomposition`` (local module)
- NumPy
- MNE (optional; enables ``DataFromMNE`` / ``MontageFromMNE`` helpers and
  ``DataToMNE`` export)
- ``BCI2000Tools`` (optional; enables richer container types and
  electrode-coordinate lookup)

Notes
-----
- The module can also be run directly as a script for quick diagnostics on
  a BCI2000 data file (see ``if __name__ == '__main__'`` block and the
  argparse help).
- Internal conventions follow the notation used in the BCI2000 / cEPOCS
  codebase: ``P`` = whitening matrix, ``R`` = rotation matrix,
  ``W = P @ R`` = combined spatial filter, ``A = Sigma @ W`` = spatial pattern.
"""

__all__ = [
	'ApplySpatialFilters',
	'Covariance',
	'Check',
	'DataFromMNE',
	'MontageFromMNE',
	'SpatialWhiteningDecomposition',
]

import os
import sys

HERE = os.path.abspath( os.path.dirname( __file__ ) ).replace( os.path.sep, '/' )
DEFAULT_FILE = HERE + '/../../cEPOCS/data/C-EPOCS-Sub01/Session1-calibration/EEG-files/20231031-1525-RRP_DSI_HC02-CT6-R16.dat'
# /Users/hill/Dropbox/bci/data/dsroot/eeg/smr/DoC/MotorImagery/cs05/raw_jez2000/CS05-MotorImagery-S001R02.dat


if __name__ == '__main__':
	def BandLimits( x ):
		import ast, numpy
		if isinstance( x, str ): x = ast.literal_eval( x )
		x = numpy.array( x, dtype=float ).ravel().tolist()
		if len( x ) not in [ 0, 1, 2 ]: raise ValueError( 'blah' )
		return x
	def TrainingSubsetSpecification( x ):
		import ast, numpy
		if isinstance( x, str ): x = ast.literal_eval( x )
		if isinstance( x, int ): x = numpy.arange( x )
		x = numpy.array( x )
		if not numpy.issubdtype( x.dtype, numpy.integer ) and x.dtype != bool: raise ValueError( 'blah' )
		return x
	import argparse
	class HelpFormatter( argparse.RawDescriptionHelpFormatter ): pass
	#class HelpFormatter( argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter ): pass
	parser = argparse.ArgumentParser( description=__doc__, formatter_class=HelpFormatter, )#   prog='python -m Whitening', )
	parser.add_argument(       "filename",  nargs='?',      default=DEFAULT_FILE,   help='BCI2000 file (DSI24 with TRG triggers)' )
	parser.add_argument( "-f", "--filterBand", metavar='BAND', default=[0.5, 8], type=BandLimits, help='filterBand argument for EpochSet' )
	parser.add_argument( "-b", "--lookBackMsec", metavar='MSEC', default=100, type=float, help='look back this many milliseconds from the TRG trigger - TODO: unfortunately you cannot skip ahead, negative values will not work here' )
	parser.add_argument( "-l", "--lookAheadMsec", metavar='MSEC', default=800, type=float, help='look ahead this many milliseconds from the TRG trigger' )
	parser.add_argument( "-r", "--maxRank", metavar='RANK', default=None, type=int, help='maximum rank to allow in sensorCovariance decomposition' )
	parser.add_argument( "-t", "--trainingSubset", metavar='INDICES_OR_MASK', default=range( 40 ), type=TrainingSubsetSpecification, help='trainingSubset argument for chosen method' )
	parser.add_argument( "-m", "--method", metavar='METHOD_NAME', default='XDAWN', help='name of SpatialWhiteningDecomposition method that wraps .Rayleigh() is the particular way suited for your dataset' )
	
	OPTS = parser.parse_args()

import numpy

from SVD import SingularValueDecomposition

BaseClass = object
Container = dict
ChannelSet = None

try: from BCI2000Tools.Container import Bunch as Container, Bunch as BaseClass # nice-to-have, but not required
except: pass
else: Container._summarize = 80
try: from BCI2000Tools.Electrodes import ChannelSet # nice-to-have, but not required
except: pass

def ApplySpatialFilters( signal, spatialFilteringMatrix, sensorAxis=1 ):
	if signal is None: return None
	if spatialFilteringMatrix.ndim == 1: spatialFilteringMatrix = spatialFilteringMatrix[ :, None ]
	signalLabels = ''.join( chr( 97 + i ) for i in range( signal.ndim ) )
	sensorLabel = signalLabels[ sensorAxis ]
	outputChannelLabel = sensorLabel.upper()
	spatialFilterLabels = sensorLabel + outputChannelLabel
	outputLabels = signalLabels.replace( sensorLabel, outputChannelLabel )
	sub = signalLabels + ',' + spatialFilterLabels + '->' + outputLabels
	return numpy.einsum( sub, signal, spatialFilteringMatrix )

def Covariance( x, preservedAxis=-1 ):
	try: x, preservedAxis, mneObject = DataFromMNE( x )
	except: pass
	
	denominator = x.size / x.shape[ preservedAxis ]
	labels = ''.join( chr( 97 + i ) for i in range( x.ndim ) )
	outputRowLabel = labels[ preservedAxis ]
	outputColumnLabel = outputRowLabel.upper()
	sub = labels + ',' + labels.replace( outputRowLabel, outputColumnLabel ) + '->' + outputRowLabel + outputColumnLabel
	out = numpy.einsum( sub, x, x )
	out /= denominator
	return out

def Select( x, selection, axis, keepdims=False ):
	"""
	Returns `x[ ..., :, :, selection, :, :, ... ]` with `selection` in the position
	specified by `axis`.
	
	Example: if `x.dim` is 3, `Select(x, sel, axis=1)` returns `x[:, sel, :]`.
	
	Unlike `numpy.take` or `numpy.take_along_axis`,  `selection` may be anything that
	works as a subscript (integer, sequence of integers, boolean mask sequence) and
	will produce the corresponding behavior in all cases.  As a special case, it can
	also be `None`, which is taken to mean the same as `slice(None)`, i.e. all of `x`.
	
	You can specify `axis` as a tuple or list, in which case `selection` must be a
	same-length tuple or list of corresponding selections: the slicing is then
	performed iteratively across the specified axes (the subscripts are *not*
	advanced in lock-step) - for example, again when `x.ndim` is 3::
	
		Select(x, [sel2, sel0], axis=[2,0])  ->  x[:, :, sel2][sel0, :, :]

	or, if `sel2` is an integer (thereby reducing dimensionality when applied as
	a subscript)::

		Select(x, [sel2, sel0], axis=[2,0])  ->  x[:, :, sel2][sel0, :]
	
	You can pass `keepdims=True` to prevent integer subscripts from reducing
	dimensionality (then any integer `selection` argument gets treated as a
	one-element list, `[selection]`).
	
	"""
	if not isinstance( axis, ( tuple, list, numpy.ndarray ) ): selection, axis = [ selection ], [ axis ]
	if not hasattr( selection, '__len__' ): selection = [ selection ] # catches the edge case where `axis` was specified as a sequence but `selection` was an integer
	axis = numpy.asarray( axis, dtype=int ).ravel()
	axis[ axis < 0 ] += x.ndim
	if max( axis ) >= x.ndim: raise ValueError( 'there is no axis %d' % max( axis ) ) 
	if min( axis ) < 0: raise ValueError( 'axis too negative' ) 
	if len( numpy.unique( axis ) ) != len( axis ): raise ValueError( 'same axis addressed more than once' )
	if len( axis ) != len( selection ): raise ValueError( 'number of selections must match number of axes' )
	out = x
	axes = list( range( x.ndim ) )
	for selection, axis in zip( selection, axis ):
		if selection is None: continue
		if isinstance( selection, bool ): raise TypeError( 'selections cannot be scalar booleans' ) # numpy actually allows this, but it produces very unexpected dimensionality-increasing results (its inherent semantics are very obscure; if it ever had a use-case, its use there is now superseded by numpy.newaxis)
		if isinstance( selection, numpy.ndarray ):
			removeAxis = ( selection.ndim == 0 )
		else:
			try: int( selection )
			except: removeAxis = False
			else:   removeAxis = True
		if removeAxis and keepdims: selection = [ selection ]; removeAxis = False
		out = out[ tuple( selection if a == axis else slice( None ) for a in axes ) ]
		if removeAxis: axes.remove( int( axis ) )
	return out
	
def Check( x, against=None, name='' ):
	x = numpy.asarray( x )
	if against is None:
		dim = numpy.asarray( x.shape ).min()
		print( 'checking %s%sagainst eye(%d)' % ( name and ' ', name, dim ) )
		against = numpy.eye( dim )
	maxAbsDiff = float( numpy.abs( x - against ).ravel().max() )
	return maxAbsDiff


def DataToMNE( signal, samplesPerSecond, channels=None, sensorAxis=1, epochAxis=None, lookBackMsec=0.0, epochLabels=None, labelNames=None ):
	import mne
	if channels is not None:
		if hasattr( channels, 'get_labels' ): channels = channels.get_labels()
		if isinstance( channels, str ): channels = channels.replace( ',', ' ' ).split()
		channels = list( channels )
	info = mne.create_info( ch_names=channels, sfreq=samplesPerSecond, ch_types='eeg' )
	# ch_types may also be a list like ['eeg', 'eeg', 'stim']
	if signal.ndim == 2:
		if epochAxis is not None: raise ValueError( 'for 2-D data, epochAxis must be None' )
		signal = numpy.asarray( signal, dtype=float ).swapaxes( 0, sensorAxis )
		return mne.io.RawArray( data, info )
	if signal.ndim == 3:
		if epochAxis is None: epochAxis = 0; # raise ValueError( 'for 3-D data, epochAxis cannot be None' )
		if epochAxis  < 0: epochAxis  += signal.ndim
		if sensorAxis < 0: sensorAxis += signal.ndim
		if epochAxis == sensorAxis: raise ValueError( 'epochAxis and sensorAxis cannot refer to the same axis' )
		axisOrder = [ epochAxis, sensorAxis ]
		axisOrder += [ axis for axis in range( signal.ndim ) if axis not in axisOrder ]
		signal = numpy.asarray( signal, dtype=float ).transpose( axisOrder )
		nEpochs = signal.shape[ 0 ]
		if epochLabels is None: epochLabels = numpy.zeros( nEpochs )
		epochLabels = numpy.asarray( epochLabels, dtype=int ).ravel()
		if labelNames is None: labelNames = {}
		labelNames = dict( labelNames )
		for code in numpy.unique( epochLabels ):
			if code not in labelNames.values():
				labelNames[ 'trialType_%d' % code ] = code
		return mne.EpochsArray( signal, info, verbose=False,
			events = numpy.column_stack( [ numpy.arange( nEpochs ), numpy.zeros( nEpochs, int ), epochLabels ] ),
			event_id = labelNames,
			tmin = -abs( lookBackMsec / 1000.0 ),
		)
		raise NotImplementedError
		
def DataFromMNE( mneObject, picks=( 'eeg', 'meg' ) ):
	"""
	returns ( dataArray, sensorAxis, mneObject )
	"""
	if isinstance( mneObject, str ): mneObject = mne.io.read_raw( mneObject, verbose=False )
	data = mneObject.get_data( picks=picks )
	if data.ndim == 2: data = data.T; sensorAxis = 1  #  time  x sensors
	if data.ndim == 3:                sensorAxis = 1  # epochs x sensors x time
	return data, sensorAxis, mneObject
	
def MontageFromMNE( mneObject ):
	montage = mneObject.get_montage() # 3D coords are in a dict at montage.get_position()[ 'ch_pos' ]
	if ChannelSet is None: return montage.ch_names
	import mne.channels
	try:
		coords_2d = numpy.round( mne.channels.make_eeg_layout( mneObject.info ).pos[ :, :2 ], 2 )
	except:
		print( 'failed to get sensor coordinates' ) # TODO: fallback might involve looking up locations by standard names
		return ChannelSet( montage.ch_names )
	coords_2d -= [ [ 0.485, 0.5 ] ]; coords_2d *= 2.4; # TODO: might need to work on this; these numbers might not be one-size-fits-all
	#coords_2d -= coords_2d.mean( axis=0, keepdims=True )
	return ChannelSet( '\n'.join( '%s %g %g' % ( name, x, y ) for name, ( x, y ) in zip( montage.ch_names, coords_2d ) ) )
	


def alias( targetName ):
	def getter(	self ): return getattr( self, targetName )
	def setter( self, value ): setattr( self, targetName, value )
	return property( fget=getter, fset=setter )
def add_aliases( cls ):
	for name, target in getattr( cls, '_aliases', {} ).items():
		setattr( cls, name, alias( target ) )
	return cls

@add_aliases
class SpatialWhiteningDecomposition( BaseClass ):
	_summarize = 80
	_aliases = dict(
		X = 'mixedSignals',             # shape is interpreted according to sensorAxis (default: second axis)
		s = 'nSensors',
		t = 'nSamples',
		u = 'nSources',
		I_u = 'identity',
		Sigma = 'sensorCovariance',
		U = 'whitenedSignals',
		P = 'spatialWhiteningMatrix',   # each COLUMN is a spatial filter
		R = 'spatialRotationMatrix',    # each COLUMN corresponds to one source
		W = 'spatialFilteringMatrix',   # each COLUMN is a spatial filter
		A = 'spatialPatternMatrix',     # each COLUMN is a spatial pattern
		Z = 'sourceSignals',
	)
	def __init__( self, mixedSignals=None, channels=None, keepChannels=None, dropChannels=None, sensorAxis=1, maxRank=None, sensorCovariance=None ):
	
		nChannelsInfo = {}
		
		if mixedSignals is None and sensorCovariance is None:
			raise ValueError( 'if mixedSignals is not supplied, sensorCovariance must be' )
		if sensorCovariance is not None:
			referenceField = 'sensorCovariance'
			sensorCovariance = numpy.asarray( sensorCovariance, dtype=float )
			nChannels = nChannelsInfo[ 'sensorCovariance' ] = sensorCovariance.shape[ 0 ]
			if list( sensorCovariance.shape ) != [ nChannels ] * 2: raise ValueError( 'sensorCovariance should be a square 2-D array' )
		if mixedSignals is not None:
			try: mixedSignals, sensorAxis, mneObject = DataFromMNE( mixedSignals )
			except: pass
			else:
				if channels is None: channels = MontageFromMNE( mneObject )
			referenceField = 'mixedSignals'
			mixedSignals = numpy.asarray( mixedSignals, dtype=float )
			nChannels = nChannelsInfo[ 'mixedSignals' ] = mixedSignals.shape[ sensorAxis ]
			
		if isinstance( keepChannels, ( numpy.ndarray, tuple, list ) ) and numpy.array( keepChannels ).ndim == 1 and numpy.array( keepChannels ).dtype == bool: nChannelsInfo[ 'keepChannels' ] = len( keepChannels )
		if isinstance( dropChannels, ( numpy.ndarray, tuple, list ) ) and numpy.array( dropChannels ).ndim == 1 and numpy.array( dropChannels ).dtype == bool: nChannelsInfo[ 'dropChannels' ] = len( dropChannels ) 
		if channels is not None:
			if ChannelSet is None:
				if isinstance( channels, str ): channels = channels.replace( ',', ' ' ).split()
				else: channels = numpy.asarray( channels, dtype=object ).ravel().tolist()
			else:
				channels = ChannelSet( channels )
			nChannelsInfo[ 'channels' ] = len( channels )
		for key, value in nChannelsInfo.items():
			if value != nChannels: raise ValueError( 'number of channels indicated by `%s` (%d) does not match number of channels indicated by `%s` (%d)' % ( key, value, referenceField, nChannels ) )	
			
		if isinstance( keepChannels, str ):
			if not keepChannels: keepChannels = None
			elif channels is None: raise ValueError( 'to map names in keepChannels to indices, supply a ChannelSet instance or list of channel names as the `channels` argument' )
			else: keepChannels = [ channelIndex for channelIndex in self.Find( keepChannels, channels, alwaysList=True ) if channelIndex is not None ]
		if isinstance( dropChannels, str ):
			if not dropChannels: dropChannels = None
			elif channels is None: raise ValueError( 'to map names in dropChannels to indices, supply a ChannelSet instance or list of channel names as the `channels` argument' )
			else: dropChannels = [ channelIndex for channelIndex in self.Find( dropChannels, channels, alwaysList=True ) if channelIndex is not None ]
		if keepChannels is None: keepChannels = numpy.ones(  nChannels, dtype=bool )
		if dropChannels is None: dropChannels = numpy.zeros( nChannels, dtype=bool )
		keepChannels = numpy.array( keepChannels )
		dropChannels = numpy.array( dropChannels )
		if keepChannels.dtype.name.startswith( ( 'int', 'uint' ) ): channelIndices = keepChannels; keepChannels = numpy.zeros( nChannels, dtype=bool ); keepChannels[ channelIndices ] = True
		if dropChannels.dtype.name.startswith( ( 'int', 'uint' ) ): channelIndices = dropChannels; dropChannels = numpy.zeros( nChannels, dtype=bool ); dropChannels[ channelIndices ] = True
		
		if channels is None:
			channels = [ 'Ch%03d' % ( i + 1 ) for i in range( nChannels ) ]
			if ChannelSet is not None: channels = ChannelSet( channels )
		self.allChannels = channels
		self.keepChannels = keepChannels & ~dropChannels
		if channels is not None:
			if isinstance( channels, list ): channels = [ channel for channel, keep in zip( channels, self.keepChannels ) if keep ]
			else: channels = channels[ self.keepChannels ]
		self.channels = channels
		self.sensorAxis = sensorAxis
		self.nSensors = None
		self.sensorCovariance = None
		self.mixedSignals = None
		self.nSamples = None
		if sensorCovariance is not None:
			self.sensorCovariance = sensorCovariance[ self.keepChannels, : ][ :, self.keepChannels ]
			self.nSensors = self.sensorCovariance.shape[ 0 ]
		if mixedSignals is not None:
			if self.sensorAxis < 0: self.sensorAxis += mixedSignals.ndim
			self.mixedSignals = mixedSignals[ tuple( self.keepChannels if axis == self.sensorAxis else slice( None ) for axis in range( mixedSignals.ndim ) ) ]
			self.nSensors = self.mixedSignals.shape[ self.sensorAxis ]
			self.nSamples = self.mixedSignals.size / self.nSensors
		if self.sensorCovariance is None:
			self.sensorCovariance = Covariance( self.mixedSignals, self.sensorAxis )
			
		self.sigmaSVD = None
		self.spatialWhiteningMatrix = None
		self.nSources = None
		self.identity = None
		self.SetRotation( None )
		
		self.Whiten( maxRank )

	def Whiten( self, maxRank=None ):
		if maxRank in [ 'max', 'full', 'auto' ]: r = None
		if self.sigmaSVD is None:
			self.sigmaSVD = SingularValueDecomposition( self.sensorCovariance, maxRank=maxRank, hermitian=True )
		else:
			self.sigmaSVD.imposedRank = maxRank # decomposition is already done and stored, so here we just change the number of columns we're using
		self.spatialWhiteningMatrix = self.sigmaSVD.whitener
		self.nSources = self.spatialWhiteningMatrix.shape[ 1 ]
		self.identity = numpy.eye( self.nSources )
		rayleigh = self.unwhitenedRayleighMatrix
		self.SetRotation( None )
		if rayleigh is not None: self.Rayleigh( rayleigh, alreadyWhitened=False )
	
	def SetRotation( self, R ):
		if R is not None:
			R = numpy.array( R, dtype=float )
			expectedShape = ( self.nSources, self.nSources )
			if tuple( R.shape ) != expectedShape: raise ValueError( 'rotation matrix must be %d-by-%d' % expectedShape )
			if Check( R.dot( R.T ) ) > 1e-10:     raise ValueError( 'not a rotation matrix' )
		self.spatialRotationMatrix    = R
		self.spatialFilteringMatrix   = None if R is None else self.spatialWhiteningMatrix.dot( self.spatialRotationMatrix )
		self.spatialPatternMatrix     = None if R is None else self.sensorCovariance.dot( self.spatialFilteringMatrix )
		self.sourceSignals            = None if R is None else ApplySpatialFilters( self.mixedSignals, self.spatialFilteringMatrix, self.sensorAxis )
		self.eigenvalues              = None
		self.unwhitenedRayleighMatrix = None
		self.whitenedRayleighMatrix   = None
		return self
	
	def Find( self, labels, channelSet=None, alwaysList=False ):
		if isinstance( labels, str ): labels = labels.replace( ',', ' ' ).split(); inputAsSequence = len( labels ) > 1
		else: inputAsSequence = isinstance( labels, ( tuple, list, numpy.ndarray ) ); labels = numpy.asarray( labels, dtype=object ).ravel().tolist()
		if channelSet is None: channelSet = self.channels
		if channelSet is None: channelSet = []
		if hasattr( channelSet, 'find_labels' ): # duck-detects BCI2000Tools.Electrodes.ChannelSet instance
			indices = channelSet.find_labels( labels )
		else:
			def Resolve( label ):
				if isinstance( label, ( int, float ) ): return int( label ) if 0 <= label < len( channelSet ) else None
				try: return channelSet.index( label )
				except ValueError: return None
			indices = [ Resolve( label ) for label in labels ]
		if not inputAsSequence and len( indices ) == 1 and not alwaysList: indices = indices[ 0 ]
		return indices
	
	@property
	def whitenedSignals( self ): # generally, we will not need this
		return ApplySpatialFilters( self.mixedSignals, self.spatialWhiteningMatrix, self.sensorAxis )
	
	@property
	def check( self ):
		d = Container( u=self.nSources )
		d[ 'P.T @ Sigma @ P'   ] = None if self.P is None else Check( self.P.T.dot( self.Sigma ).dot( self.P ),   self.I_u )
		d[ 'U.T @ U / t'       ] = None if self.X is None else Check( Covariance( self.U, self.sensorAxis ),      self.I_u )
		d[ 'R.T @ R'           ] = None if self.R is None else Check( self.R.T.dot( self.R ),                     self.I_u )
		d[ 'W.T @ Sigma @ W'   ] = None if self.W is None else Check( self.W.T.dot( self.Sigma ).dot( self.W ),   self.I_u )
		d[ 'W.T @ A'           ] = None if self.W is None else Check( self.W.T.dot( self.A ),                     self.I_u )
		d[ 'Z.T @ Z / t'       ] = None if self.Z is None else Check( Covariance( self.Z, self.sensorAxis ),      self.I_u )
		d[ 'A @ A.T vs Sigma*' ] = None if self.A is None else Check( self.A.dot( self.A.T ), self.sigmaSVD.reconstruction ) # A@A.T can be equal self.Sigma, even if Sigma is rank-deficient---but it will not equal self.Sigma if we have used maxRank to artificially further reduce the rank of the *effective* Sigma that was used to compute A)
		return d
	
	def Rayleigh( self, H, alreadyWhitened=False ):
		"""
		Rayleigh quotient optimization.
		
		      argmax(w) of (    w.T    @ H @   w   ) / (    w.T    @ Sigma @   w   )
		= P @ argmax(r) of ( r.T @ P.T @ H @ P @ r ) / ( r.T @ P.T @ Sigma @ P @ r )
		= P @ argmax(r) of ( r.T @ P.T @ H @ P @ r ) / ( r.T           @         r )
				
		"""
		
		try: H, sensorAxis, mneObject = DataFromMNE( H )
		except: pass
		else: H = Covariance( H[ :, self.keepChannels ], sensorAxis ); alreadyWhitened=False
		
		expectedShape = [ self.nSources if alreadyWhitened else self.nSensors ] * 2
		if list( H.shape ) != expectedShape: raise ValueError( 'matrix shape is expected to be %r%s' % ( expectedShape, '' if self.nSensors == self.nSources else ' when alreadyWhitened=%r' % bool( alreadyWhitened ) ) )
		if Check( H, H.conj().T ) > 1e-12: raise( 'matrix is expected to be %s' % ( 'symmetric' if numpy.isrealobj( H ) else 'Hermitian' ) )
		if alreadyWhitened:
			unwhitenedRayleighMatrix = None
			whitenedRayleighMatrix = H
		else:
			unwhitenedRayleighMatrix = H
			whitenedRayleighMatrix = self.spatialWhiteningMatrix.T.dot( H ).dot( self.spatialWhiteningMatrix )
			# NB: formulation as a generalized eigenvalue problem is probably not the way to go:
			#                   eigenvalues, W = scipy.linalg.eigh( H, self.sensorCovariance )        # throws an exception when sensorCovariance is not full rank
			#                   eigenvalues, W = scipy.linalg.eigh( H, self.sigmaSVD.reconstruction ) # ...which you can explore by restricting maxRank and doing this
			# Even if the algorithm were to succeed, it would gives you W directly, and if you want to solve for R you have to find a matrix Q
			# such that R = Q @ W. Unfortunately if that transformation is a tall matrix, W = P @ R won't then get you back to the W that scipy gave you.
                
		eigenvalues, R = numpy.linalg.eigh( whitenedRayleighMatrix )
		# eigh() delivers orthonormal columns, which is what we want, but it delivers eigenvalues and eigenvectors in ascending eigenvalue order
		self.SetRotation( R[ :, ::-1 ].copy() )        # so let's reverse the order of the eigenvectors
		self.eigenvalues = eigenvalues[ ::-1 ].copy()  # and let's reverse the order of the eigenvalues
		self.unwhitenedRayleighMatrix = unwhitenedRayleighMatrix
		self.whitenedRayleighMatrix   = whitenedRayleighMatrix
		return self

	def PlotEigs( self, normalized=False, hold=False, marker='o', axes=None, **kwargs ):
		eigs = self.eigenvalues
		if normalized: eigs = eigs / sum( eigs )
		call_ion = 'IPython' in sys.modules and 'matplotlib' not in sys.modules
		import matplotlib.pyplot as plt; call_ion and plt.ion()
		if axes is None: axes = plt.gca()
		if not hold: axes.cla()
		axes.plot( eigs, marker=marker, **kwargs )
		axes.set_ylim( [ 0, max( axes.get_ylim() ) ] )
		
	def PlotEpochs( self, channel=None, projectBack=None, componentIndex=None, mean=False, axes=None, figure=None, timebase=None, raster=False, xlim=None, ylim=None, **kwargs ):
		"""
		self.PlotEpochs(channel='Cz', projectBack=None)  # plot the original Cz mixture
		self.PlotEpochs(channel='Cz', projectBack=0)     # plot the projection of the first source (index 0) at Cz
		self.PlotEpochs(channel='Cz', projectBack=1)     # plot the projection of the second source (index 1) at Cz
		self.PlotEpochs(channel='Cz', projectBack=[0,1]) # plot the projection of the first two sources at Cz
		self.PlotEpochs(channel='Cz', projectBack=range(5)) # plot the projection of the first five sources at Cz
		self.PlotEpochs(componentIndex=0) # plot the first source signal (index 0) in the source space (note that the sign may be arbitrarily flipped)
		"""
		# TODO: baseline subtraction option
		if isinstance( projectBack, ( tuple, list ) ) and not projectBack: projectBack = None
		if componentIndex is None and channel is None:
			if projectBack is None: projectBack = 1
			for candidate in [ 'Cz', 'Pz', 'FCz', 'CPz', 'Fz', 'POz', 0 ]:
				if self.Find( candidate ): channel = candidate; break
		if componentIndex is not None:
			if channel     is not None: raise ValueError( 'cannot use componentIndex and channel at the same time' )
			if projectBack is not None: raise ValueError( 'cannot use componentIndex and projectBack at the same time' )
			signal = self.sourceSignals[ :, componentIndex, : ].T
			kwargs.setdefault( 'title', 'Source component #%d' % componentIndex )
		mixingMatrix = self.spatialPatternMatrix
		if channel is not None:
			if projectBack is not None:
				projectBack = numpy.asarray( projectBack ).ravel()
				nProjected = mixingMatrix[ :, projectBack ].shape[ 1 ]
				if nProjected == self.sigmaSVD.rank: projectBack = None
			if projectBack is None:
				signal = self.mixedSignals
				kwargs.setdefault( 'title', '%s (original full mixture)' % channel )
			else:
				signal = ApplySpatialFilters( self.sourceSignals[ :, projectBack, : ], mixingMatrix[ :, projectBack ].T, sensorAxis=self.sensorAxis )
				kwargs.setdefault( 'title', '%s (cleaned: %d projected source%s)' % ( channel, nProjected, '' if nProjected == 1 else 's' ) )
			signal = signal[ :, self.Find( channel ), : ].T
			
		if raster:
			from BCI2000Tools.Plotting import imagesc # TODO - would be nice not to need this
			kwargs.setdefault( 'balance', 0.0 )
			kwargs.setdefault( 'aspect', 'auto' )
			kwargs.setdefault( 'xlabel', 'Time' )
			kwargs.setdefault( 'ylabel', 'Trial Number' )
			kwargs.setdefault( 'colorbartitle', r'$\mu$V' if componentIndex is None else 'a.u.' )
			h = imagesc( signal.T, x=timebase, figure=figure, axes=axes, **kwargs )
			if xlim is not None: h.axes.set_xlim( xlim )
			if ylim is not None: h.axes.set_ylim( ylim )
		else:
			call_ion = 'IPython' in sys.modules and 'matplotlib' not in sys.modules
			import matplotlib.pyplot as plt; call_ion and plt.ion()
			if axes is None: axes = figure
			if axes is None: axes = plt.gcf()
			if isinstance( axes, int ): axes = plt.figure( axes )
			if isinstance( axes, plt.Figure ): axes = axes.gca()
			title = kwargs.pop( 'title' )
			xlabel = kwargs.pop( 'xlabel', 'Time (samples)' if timebase is None else 'Time' )
			ylabel = kwargs.pop( 'ylabel', 'Amplitude (%s)' % ( r'$\mu$V' if componentIndex is None else 'arbitrary units' ) )
			axes.cla()
			if mean: signal = signal.mean( axis=1 )
			axes.plot( signal, **kwargs ) if timebase is None else axes.plot( timebase, signal, **kwargs )
			axes.set( title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim )			
	
	def CSP( self, labels, epochAxis=0, trainingSubset=None, targetLabel=None ):
		signals = self.mixedSignals
		labels = numpy.asarray( labels, dtype=int ).ravel()
		if labels.size != signals.shape[ epochAxis ]: raise( 'number of labels (%d) does not match number of epochs (%d)' % ( labels.size, signals.shape[ epochAxis ] ) )
		uniqueLabels = numpy.unique( labels )
		if len( uniqueLabels ) == 0: raise ValueError( 'no labels' ) 
		if len( uniqueLabels ) == 1: raise ValueError( 'labels specify only one class' ) 
		if targetLabel is None: targetLabel = int( uniqueLabels[ 0 ] )
		if not (labels == targetLabel ).sum(): raise ValueError( 'label %r does not appear' % targetLabel )
		print( 'Dataset has %d classes %r - targetting class label %r' % ( len( uniqueLabels ), uniqueLabels.tolist(), targetLabel ) )
		if trainingSubset is not None: # can be a sequence of indices, or a boolean mask of the correct length
			labels  = Select( labels,  trainingSubset, axis=0,         keepdims=True )
			signals = Select( signals, trainingSubset, axis=epochAxis, keepdims=True )
			if not ( labels == targetLabel ).sum(): raise ValueError( 'label %r does not appear in the training subset' % targetLabel )
			signals = Select( signals, labels == targetLabel, axis=epochAxis, keepdims=True )		
		self.Rayleigh( Covariance( signals, self.sensorAxis ), alreadyWhitened=False )
		return self

	def XDAWN( self, epochAxis=0, trainingSubset=None ):
		signals = self.mixedSignals
		if trainingSubset is not None: # can be a sequence of indices, or a boolean mask of the correct length
			signals = Select( signals, trainingSubset, axis=epochAxis, keepdims=True )
		averaged = signals.mean( axis=epochAxis, keepdims=True )
		self.Rayleigh( Covariance( averaged, self.sensorAxis ), alreadyWhitened=False )
		return self

	def XDAWN_via_mne( self, samplesPerSecond, epochAxis=0, trainingSubset=None, nComponents=4, trialType='trialType_0' ):
		# TODO: investigate why this is yielding totally different results and doesn't appear to be whitening the same covariance matrix
		import mne.preprocessing
		x = Select( self.X, trainingSubset, epochAxis )
		etr = DataToMNE( x,      samplesPerSecond=samplesPerSecond, channels=self.channels, epochAxis=epochAxis, sensorAxis=self.sensorAxis )
		ets = DataToMNE( self.X, samplesPerSecond=samplesPerSecond, channels=self.channels, epochAxis=epochAxis, sensorAxis=self.sensorAxis )
		xd = mne.preprocessing.Xdawn( n_components=nComponents )
		xd.fit( etr )
		out = xd.apply( ets )[ trialType ]
		self.SetRotation( None )
		#  filters output from mne.preprocessing.xdawn._fit_xdawn: "Each row corresponds to one component."		
		self.spatialFilteringMatrix = xd.filters_[ trialType ].T[ :, :self.nSources ]
		self.spatialPatternMatrix = xd.patterns_[ trialType ].T[ :, :self.nSources ]
		self.sourceSignals, _, _ = DataFromMNE( out )
		self.sourceSignals = Select( self.sourceSignals, range( self.nSources ), self.sensorAxis )
		self.mne_xdawn = xd
		return self

	def SSA(self, epochAxis=0, trainingSubset=None):
		"""
		Stationary Subspace Analysis (SSA) using a Rayleigh-quotient
		formulation based on epoch-wise mean shifts

		This implementation:

			- Assumes self.mixedSignals is 3-D: (epochs, sensors, time),
			  up to permutation of axes.
			- Reorders axes into (N, C, T) = (epochs, sensors, time)
			- Optionally restricts to a training subset of epochs
			- Computes epoch-wise sensor means μ_k
			- Builds a between-epoch scatter matrix

					H = sum_k (μ_k - μ)(μ_k - μ)^T,

			  where μ is the global mean across epochs
			- Calls self.Rayleigh(H), which internally whitens H and
			  solves the Rayleigh quotient

					max_w (w^T H w) / (w^T Σ w),

			  with Σ = sensor covariance

		Result:

			- Components with *larger* eigenvalues are more
			  nonstationary in their mean across epochs
			- Components with smaller eigenvalues are more stationary

		Parameters
		----------
		epochAxis : int
			Axis indexing epochs in self.mixedSignals
		trainingSubset : index-like or None
			Optional subset of epochs for fitting SSA (indices or
			boolean mask), see Select()
		"""
		import numpy

		signals = self.mixedSignals
		if signals is None:
			raise ValueError("mixedSignals is None; SSA requires epoch-based data")

		if signals.ndim != 3:
			raise ValueError(
				"SSA expects 3-D data (epochs, sensors, time); got shape %r"
				% (signals.shape,)
			)

		# 1) Put data into (N, C, T): epochs x sensors x time
		epoch_axis = int(epochAxis)
		if epoch_axis < 0:
			epoch_axis += signals.ndim

		sensor_axis = int(self.sensorAxis)
		if sensor_axis < 0:
			sensor_axis += signals.ndim

		if epoch_axis == sensor_axis:
			raise ValueError("epochAxis and sensorAxis must refer to different axes (got both = %d)" % epoch_axis )

		axis_order = [epoch_axis, sensor_axis]
		axis_order += [ax for ax in range(signals.ndim) if ax not in axis_order]
		x = numpy.transpose(signals, axes=axis_order)  # (N, D, T)

		# Optional restriction to a subset of epochs
		if trainingSubset is not None:
			x = Select(x, trainingSubset, axis=0, keepdims=True)

		N, C, T = x.shape
		if N <= 1:
			raise ValueError("SSA needs at least two epochs (got N=%d after trainingSubset)" % N )
		if T <= 0:
			raise ValueError("Each epoch must contain at least one time sample")

		# 2) Epoch-wise means and between-epoch scatter of means
		# μ_k: mean over time for each epoch, shape (N, C)
		mu = x.mean(axis=2)                      # (N, C)
		mu_bar = mu.mean(axis=0, keepdims=True)  # (1, C)
		mu_dev = mu - mu_bar                     # (N, C)

		# Between-epoch scatter:
		#   H = sum_k (μ_k - μ)(μ_k - μ)^T
		# Normalization by N is irrelevant for Rayleigh quotients
		H = mu_dev.T.dot(mu_dev) / float(N)
		# Enforce symmetry numerically
		H = 0.5 * (H + H.T)

		# 3) Rayleigh optimization in sensor space
		# This solves:
		#   max_w (w^T H w) / (w^T Σ w),
		# where Σ is the sensor covariance used in whitening
		# self.Rayleigh() will:
		#   - whiten H -> P^T H P
		#   - eigen-decompose
		#   - sort eigenvectors by descending eigenvalue
		#   - set self.R, self.W, self.A, self.Z, self.eigenvalues
		self.Rayleigh(H, alreadyWhitened=False)
		return self

	def SSA_prl(self,
		epochAxis=0,
		trainingSubset=None,
		nStationary=None,
		maxIter=100,
		stepSize=1e-2,
		reg=1e-6,
		verbose=False,
	):
		"""
		Stationary Subspace Analysis (SSA) as in von Bünau et al.,
		Phys. Rev. Lett. 103, 214101 (2009)

		This implementation follows their formulation:

		- Data are globally whitened: U = P^T X, Cov(U) ~ I.
		- We split U into N epochs
		- For each epoch i we compute empirical mean μ_i and covariance Σ_i
		- We look for a d-dimensional stationary projection B_s (rows
			orthonormal, B_s ∈ R^{d x D}) that minimizes

				L = sum_i [ -log det Σ_s_i + μ_s_i^T μ_s_i ],

			where μ_s_i = B_s μ_i and Σ_s_i = B_s Σ_i B_s^T are the
			epoch-wise mean and covariance in the stationary subspace

		We optimize B_s on the Stiefel manifold (orthonormal rows) using
		gradient descent, then complete B_s to a full D x D rotation matrix R
		and call SetRotation(R)

		Parameters
		----------
		epochAxis : int
			Axis indexing epochs in `self.whitenedSignals`
		trainingSubset : index-like or None
			Optional subset of epochs used to fit the SSA projection
			Can be integer indices or a boolean mask; see `Select`
		nStationary : int or None
			Dimension d of the stationary subspace. If None, defaults to
			floor(self.nSources / 2)
		maxIter : int
			Number of gradient descent iterations
		stepSize : float
			Gradient descent step size
		reg : float
			Diagonal regularizer added to stationary covariances Σ_s_i
		verbose : bool
			If True, prints loss every 10 iterations

		After running, the object has:
			- self.R, self.W, self.A, self.Z updated
			- self.eigenvalues set to a per-component nonstationarity
				measure (larger = more nonstationary)
		"""

		import numpy

		# ------------------------------------------------------------------
		# 1) Get whitened data and put axes in (epochs, sources, time)
		# ------------------------------------------------------------------
		signals = self.whitenedSignals
		if signals is None:
			raise ValueError(
				"whitenedSignals is None; make sure `mixedSignals` is set "
				"and `Whiten()` has been called."
			)
		if signals.ndim != 3:
			raise ValueError(
				"SSA expects 3-D data (epochs, sensors/sources, time); "
				"got shape %r" % (signals.shape,)
			)

		epoch_axis = int(epochAxis)
		if epoch_axis < 0:
			epoch_axis += signals.ndim
		sensor_axis = int(self.sensorAxis)
		if sensor_axis < 0:
			sensor_axis += signals.ndim
		if epoch_axis == sensor_axis:
			raise ValueError("epochAxis and sensorAxis must refer to different axes")

		axis_order = [epoch_axis, sensor_axis]
		axis_order += [ax for ax in range(signals.ndim) if ax not in axis_order]
		u = numpy.transpose(signals, axes=axis_order)  # (N, D, T)

		# Restrict to training subset if requested
		if trainingSubset is not None:
			u = Select(u, trainingSubset, axis=0, keepdims=True)

		N, D, T = u.shape
		if N <= 0:
			raise ValueError("No epochs available for SSA (after applying trainingSubset)")
		if T <= 1:
			raise ValueError("Each epoch must contain at least two time samples")

		# Choose stationary dimensionality d
		if nStationary is None:
			d = D // 2
		else:
			d = int(nStationary)
		if not (1 <= d <= D):
			raise ValueError(
				"nStationary must satisfy 1 <= nStationary <= %d (got %r)"
				% (D, nStationary)
			)

		# ------------------------------------------------------------------
		# 2) Precompute epoch-wise means and covariances in whitened space
		# ------------------------------------------------------------------
		# u: (N, D, T)
		mu = u.mean(axis=2)            # (N, D)
		u_c = u - mu[:, :, None]       # (N, D, T), centered per epoch
		covs = numpy.einsum("ndt,net->nde", u_c, u_c) / float(T - 1)  # (N, D, D)

		# ------------------------------------------------------------------
		# 3) Initialize B_s on Stiefel manifold: d x D, orthonormal rows
		# ------------------------------------------------------------------
		B_s = numpy.zeros((d, D), dtype=float)
		B_s[:, :d] = numpy.eye(d)

		def _loss_and_grad(B_s):
			"""
			Compute KL-based loss and Euclidean gradient wrt B_s
			We use the PRL loss:

				L = sum_i [ -log det Σ_s_i + μ_s_i^T μ_s_i ]

			with μ_s_i = B_s μ_i and Σ_s_i = B_s Σ_i B_s^T.
			"""
			B_s = B_s.reshape(d, D)
			loss = 0.0
			G = numpy.zeros_like(B_s)

			for i in range(N):
				mu_i = mu[i]      # (D,)
				Sigma_i = covs[i] # (D, D)

				mu_s_i = B_s.dot(mu_i)                     # (d,)
				Sigma_s_i = B_s.dot(Sigma_i).dot(B_s.T)    # (d, d)

				# Regularize for numerical stability
				Sigma_s_i = 0.5 * (Sigma_s_i + Sigma_s_i.T)
				Sigma_s_i = Sigma_s_i + reg * numpy.eye(d)

				try:
					inv_Sigma_s_i = numpy.linalg.inv(Sigma_s_i)
					sign, logdet = numpy.linalg.slogdet(Sigma_s_i)
				except numpy.linalg.LinAlgError:
					# Fall back to pseudo-inverse and a conservative logdet
					inv_Sigma_s_i = numpy.linalg.pinv(Sigma_s_i)
					sign, logdet = 1.0, numpy.log(
						max(reg, numpy.trace(Sigma_s_i) / float(d))
					)

				if sign <= 0:
					# Extremely degenerate; penalize heavily
					logdet = numpy.log(max(reg, numpy.trace(Sigma_s_i) / float(d)))

				# PRL Eq. (2): sum_i ( -log det Σ_s_i + μ_s_i^T μ_s_i )
				loss += -logdet + mu_s_i.dot(mu_s_i)

				# Gradient wrt B_s:
				# d/dB_s [ -log det Σ_s ] = -2 * inv(Σ_s) * B_s * Σ_i
				# d/dB_s [ μ_s^T μ_s ]   =  2 * μ_s * μ_i^T
				G += -2.0 * inv_Sigma_s_i.dot(B_s).dot(Sigma_i) \
						+ 2.0 * numpy.outer(mu_s_i, mu_i)

			return loss, G

		# ------------------------------------------------------------------
		# 4) Gradient descent on Stiefel manifold for B_s
		# ------------------------------------------------------------------
		for it in range(maxIter):
			loss, G = _loss_and_grad(B_s)

			# Project gradient onto tangent space of Stiefel manifold
			# Tangent projection: G_proj = G - B_s * sym(B_s^T G)
			BtG = B_s.T.dot(G)
			sym_BtG = 0.5 * (BtG + BtG.T)
			G_proj = G - B_s.dot(sym_BtG)

			# Gradient step in tangent space
			B_s = B_s - stepSize * G_proj

			# Re-orthonormalize rows via QR on transpose
			Q, R = numpy.linalg.qr(B_s.T)
			B_s = Q.T

			if verbose and (it % 10 == 0 or it == maxIter - 1):
				print("SSA iter %4d  loss = % .6e" % (it, loss))

		# ------------------------------------------------------------------
		# 5) Complete stationary B_s to full D x D rotation matrix R_full
		# ------------------------------------------------------------------
		# B_s has orthonormal rows; its row-space projector is:
		P_proj = B_s.T.dot(B_s)  # (D, D)
		P_proj = 0.5 * (P_proj + P_proj.T)

		# Eigen-decompose the projector: eigenvalues ~1 for stationary subspace,
		# ~0 for its orthogonal complement
		eigvals_P, eigvecs_P = numpy.linalg.eigh(P_proj)
		order = numpy.argsort(eigvals_P)[::-1]  # descending
		eigvals_P = eigvals_P[order]
		eigvecs_P = eigvecs_P[:, order]

		# First d eigenvectors span the stationary subspace, remaining the complement
		Q_stat = eigvecs_P[:, :d]      # (D, d)
		Q_non  = eigvecs_P[:, d:]      # (D, D-d)

		R_full = numpy.concatenate([Q_stat, Q_non], axis=1)  # (D, D)
		# Columns of R_full are orthonormal by construction

		# Set rotation so that sources are z(t) = R_full^T * whitenedSignals(t)
		self.SetRotation(R_full)

		# ------------------------------------------------------------------
		# 6) Define a per-component nonstationarity measure as "eigenvalues"
		# ------------------------------------------------------------------
		# For each component j, measure nonstationarity using the same
		# KL-like score in 1D: sum_i [ -log var_ij + mean_ij^2 ]
		eigs = numpy.zeros(self.nSources, dtype=float)
		for j in range(self.nSources):
			w = R_full[:, j]  # direction in whitened space (D,)
			score = 0.0
			for i in range(N):
				mu_i = mu[i]
				Sigma_i = covs[i]

				mu_ij = w.dot(mu_i)
				var_ij = w.dot(Sigma_i).dot(w)
				var_ij = max(var_ij, reg)

				score += -numpy.log(var_ij) + mu_ij * mu_ij
			eigs[j] = score

		self.eigenvalues = eigs
		self.unwhitenedRayleighMatrix = None
		self.whitenedRayleighMatrix   = None
		return self


if __name__ == '__main__':

	from BCI2000Tools.AllTools import *
	
	from BCI2000Tools.EventRelated import EpochSet
	self = EpochSet( OPTS.filename, trigger='TRG', defaultReference='A1,A2', keep=True, filterBand=OPTS.filterBand, lookBackMsec=OPTS.lookBackMsec, lookAheadMsec=OPTS.lookAheadMsec )
		
	q = SpatialWhiteningDecomposition( self.epochs, channels=self.channels, keepChannels=~self.channelIsBad, dropChannels='TRG X1 X2 X3', maxRank=OPTS.maxRank )
	if   OPTS.method == 'XDAWN': q.XDAWN( epochAxis=0, trainingSubset=OPTS.trainingSubset )
	elif OPTS.method == 'CSP'  : q.CSP(   epochAxis=0, trainingSubset=OPTS.trainingSubset, labels=self.trialType )
	elif OPTS.method == 'SSA'  : q.SSA(   epochAxis=0, trainingSubset=OPTS.trainingSubset )
	else: raise ValueError( 'unsupported method %r' % OPTS.method )
	print( q.check )
	
	cmp1 = lambda s: numpy.abs( s.X - ApplySpatialFilters( s.Z, s.A.T, s.sensorAxis ) ).max() # this one will only work well if you set maxRank=None
	cmp2 = lambda s: numpy.abs( s.U - ApplySpatialFilters( s.Z, s.R.T, s.sensorAxis ) ).max()
	
	# For XDAWN:
	#     q.PlotEpochs('Cz', timebase=self.epochTimeMsec, raster=True, projectBack=None)  # original
	#     q.PlotEpochs('Cz', timebase=self.epochTimeMsec, raster=True, projectBack=0)     # XDAWNed to the max (first component only)
	
