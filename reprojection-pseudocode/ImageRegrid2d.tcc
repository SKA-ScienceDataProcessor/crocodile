//# ImageRegrid.cc: Regrids images

#ifndef IMAGES_IMAGEREGRID_TCC
#define IMAGES_IMAGEREGRID_TCC

#include <casacore/images/Images/ImageRegrid.h>

#include <casacore/casa/Arrays/ArrayAccessor.h>
#include <casacore/casa/Exceptions/Error.h>
#include <casacore/coordinates/Coordinates/DirectionCoordinate.h>
#include <casacore/coordinates/Coordinates/LinearCoordinate.h>
#include <casacore/coordinates/Coordinates/SpectralCoordinate.h>
#include <casacore/images/Images/SubImage.h>
#include <casacore/images/Images/TempImage.h>
#include <casacore/lattices/Lattices/LatticeUtilities.h>
#include <casacore/measures/Measures/MCDirection.h>
#include <casacore/measures/Measures/MCFrequency.h>
#include <casacore/scimath/Mathematics/InterpolateArray1D.h>

#include <casacore/casa/sstream.h>
#include <casacore/casa/fstream.h>

namespace casacore { //# NAMESPACE CASACORE - BEGIN
	
	
//
// Program structure:
//
//									     PUBLIC: regrid()
//
//										|	|
//		o---------------------------------------o-----------------------o	o---------------o
//		|					|						|
//													|
//	  checkAxes()			      regridOneCoordinate()					|
//													|
//						|		|					|
//		o-------------------------------o		o---------------------------------------o
//		|											|
//
//   regridTwoAxisCoordinate()								   	    findMaps()
//
//		|
//		o-----------------------o
//		|			|
//
//  make2DCoordinateGrid()	findScaleFactor()
//


#
# PUBLIC CLASS MEMBERS
#


template<class T>
ImageRegrid<T>::ImageRegrid()
{;}

template<class T>
ImageRegrid<T>::~ImageRegrid()
{;}


// Regrid 'inImage' to 'outImage' for the list of axes supplied in 'outPixelAxesU'.
// The input and output images have their own coordinate systems, and may contain more than
// two axes (i.e. ra, dec and spectral).
//
// Regridding can mean:
//
//	1. re-ordering the coordinate system axes (i.e. ra, dec, specral --> spectral, dec, ra).
//	2. scaling the image pixel axes. the map between input and output pixels (usually called in2DPos) can be build manually
//		(which just describes the scaling of image pixels from an NxN to an MxM image size) or it can be built by
//		CASA through a conversion from output pixel coordinates to world coordinates, and then a second conversion
//		back to input pixel coordinates (the CASA libraries will handle any UM conversions automatically). The latter case
//		(which is initiated by setting replicate = false) is intended for reprojection in addition to regridding.
//
// outImage:		the output image (return value)
// interpolationMethod:	the interpolation method
// outPixelAxesU:	a user-supplied list of axes which must be regridded
// inImage:		the input image
// replicate:		true = input and output axes are aligned, false = let CASA do reprojection for us.

template<class T>
void ImageRegrid<T>::regrid(	ImageInterface<T>& outImage,
				typename Interpolate2D::Method interpolationMethod,
				const IPosition& outPixelAxesU,
				const ImageInterface<T>& inImage,
				bool replicate )
{

	IPosition outShape = outImage.shape();
	IPosition inShape = inImage.shape();

	// display warning:
	// (inImage.ndim() != outImage.ndim())
	// The input and output images must have the same number of axes

	// display warning:
	// (inImage.imageInfo().hasMultipleBeams()) && (inImage.coordinates().hasSpectralAxis()) && (outPixelAxesU.asVector() == inImage.coordinates().spectralAxisNumber())
	// This image has multiple beams. The spectral axis cannot be regridded
	
	// display warning:
	// (inImage.imageInfo().hasMultipleBeams()) && (inImage.coordinates().hasPolarizationCoordinate()) && (outPixelAxesU.asVector() == inImage.coordinates().polarizationAxisNumber())
	// This image has multiple beams. The polarization axis cannot be regridded

	const CoordinateSystem& inCoords = inImage.coordinates();
	CoordinateSystem outCoords = outImage.coordinates();

	// Find world and pixel axis maps.
	// pixelAxisMap1 gives the position of each output pixel axis i in the input coordinate system.
	// pixelAxisMap2 gives the position of each input pixel axis i in the output coordinate system.
	Vector<int> pixelAxisMap1, pixelAxisMap2;
	findMaps( inImage.ndim(), pixelAxisMap1, pixelAxisMap2, inCoords, outCoords );

	// Check user pixel axes specifications
	_checkAxes( outPixelAxesU, inShape, outShape, pixelAxisMap1, outCoords );

	// Create a temporary lattice with a coordinate system that matches the input image. Each iteration of the regridder
	// will regrid from the temporary input image to the temporary output image, and this output will be used as input
	// for the next iteration.
	MaskedLattice<T>* tmpInPtr = NULL;
	CoordinateSystem tmpCoords( inCoords );
	
	// Set the initial output image to a copy of the input image. This just 'seeds' the iteration, and each pass
	// will set the next intput pointer to the last output pointer.
	MaskedLattice<T>* tmpOutPtr = inImage.cloneML();
	
	// Default the DONE flag to true for each axis in the output coordinate system. These are then updated to false
	// for each output pixel axis because these are the only ones that need to be regridded.
	Vector<bool> doneOutPixelAxes( outCoords.nPixelAxes(), true );
	for ( unsigned int i = 0; i < outPixelAxesU.nelements(); i++ )
		doneOutPixelAxes( outPixelAxesU[i] ) = false;

	// Loop over all the output pixel axes. Each iteration can regrid either one or two axes (e.g. for a spectral axis
	// and pair of direction axes respectively) and will update the DONE flag accordingly for each one. Therefore we must
	// check inside the loop to see if this axis has already been regridded by an earlier iteration. NOTE: Here, the
	// code is restricted to regridding 2-d axes only.
	for ( unsigned int i = 0; i < outPixelAxesU.nelements(); i++ )
		if (doneOutPixelAxes( outPixelAxesU[i] ) == false)
		{

			// Delete the last input pointer (if this is not the first iteration).
			if (tmpInPtr != NULL)
				delete tmpInPtr;
			
			// Set input and output images for this pass. The new input must be the last output image (i.e. the
			// temporary image which is originally set to be a copy of the input image).
			tmpInPtr = tmpOutPtr;
			tmpOutPtr = NULL;

			// Regrid one Coordinate, pertaining to this axis. If the axis belongs to a DirectionCoordinate or
			// 2-axis LinearCoordinate, it will also regrid the other axis. After the first pass,
			// the output image is in the final order.
			regridOneCoordinate(	doneOutPixelAxes, tmpInPtr, tmpOutPtr, outCoords, tmpCoords, outPixelAxesU[i],
						inImage, outShape, interpolationMethod, replicate );

			// Also need to update the temporary coordinate system to match the output coordinate system - we need
			// to do this because the axes in the output coordinate system may be in a different order than the
			// axes in the input coordinate system, and our re-gridded temporary image will now be using the output
			// image order. However, there may still be axes that haven't been regridded, and the coordinates need
			// to be replaced with coordinates from the input coordinate system.
			tmpCoords = outCoords;
			for ( unsigned int j = 0; j < doneOutPixelAxes.nelements(); j++ )

				// For every pixel axis that has not yet been regridded, put back the original coordinate
				// (so that its other axes can be regridded)
				if ( doneOutPixelAxes( j ) == false )
				{
					int tmpCoord, tmpAxisInCoord;
					tmpCoords.findPixelAxis( tmpCoord, tmpAxisInCoord, j );
					int inCoord, inAxisInCoord;
					inCoords.findPixelAxis( inCoord, inAxisInCoord, pixelAxisMap1[j] );

					// We might end up replacing the same coordinate more than once with this algorithm
					tmpCoords.replaceCoordinate( inCoords.coordinate( inCoord ), tmpCoord );
				}
		}
		
	// Copy the final temporary image to the output image. Copying pointers won't work - we need to actually duplicate the
	// data.
	MaskedLattice<T>* finalOutPtr = &outImage;
	LatticeUtilities::copyDataAndMask( _os, *finalOutPtr, *tmpOutPtr );

	// Cleanup.
	if (tmpInPtr != NULL)
		delete tmpInPtr;
	if (tmpOutPtr != NULL)
		delete tmpOutPtr;

} // ImageRegrid<T>::regrid


#
# PRIVATE CLASS MEMBERS
#


// Accepts a partially-regridded input image, and regrids over two further axes by creating a new
// output image and passing it to 'regridTwoAxisCoordinate'. Note that the input and output images can
// have more than two axes.
//
// A coordinate in CASA terminology consists of one or more related axes from a (possibly) larger coordinate system. For example,
// if we had an image with axes ra, dec and spectral then ra and dec would form one coordinate and the spectral axis would form
// another. In this version of the code we only regrid 2-d coordinates.
//
// doneOutPixelAxes:	boolean array which states whether or not each axis has been regridded yet
// inPtr:		the input image (the current partially-regridded image; updated on each iteration)
// outPtr:		the output image (return value, should initially be NULL).
// outCoords:		the output coordinate system
// inCoords:		the input coordinate system
// outPixelAxis:	the axis to be regridded
// inImage:		the original (non-regridded) input image
// outShape:		n-dimensional array giving the size of the n output image coordinates
// interpolationMethod:	the interpolation method to use
// replicate:		true = input and output axes are aligned, false = let CASA do reprojection for us.

template<class T>
void ImageRegrid<T>::regridOneCoordinate(	Vector<bool>& doneOutPixelAxes,
						MaskedLattice<T>* &inPtr,
						MaskedLattice<T>* &outPtr,
						CoordinateSystem& outCoords,
						const CoordinateSystem& inCoords,
						int outPixelAxis,
						const ImageInterface<T>& inImage,
						const IPosition& outShape,
						typename Interpolate2D::Method interpolationMethod,
						bool replicate )
{

	// Find world and pixel axis maps for the input and output coordinate systems.
	Vector<int> pixelAxisMap1, pixelAxisMap2;
	findMaps( inImage.ndim(), pixelAxisMap1, pixelAxisMap2, inCoords, outCoords );

	int outCoordinate, outAxisInCoordinate, inCoordinate, inAxisInCoordinate;
	
	// find the output pixel coordinate and axis ID.
	outCoords.findPixelAxis( outCoordinate, outAxisInCoordinate, outPixelAxis );
	Coordinate::Type type = outCoords.type( outCoordinate );

	// Find equivalent Coordinate in input image.
	int inPixelAxis = pixelAxisMap1[outPixelAxis];
	inCoords.findPixelAxis( inCoordinate, inAxisInCoordinate, inPixelAxis );
	
	// display warning:
	// condition: (inCoordinate == -1 || inAxisInCoordinate == -1)
	// message: Output axis (outPixelAxis + 1) of coordinate type (outCoords.showType(outCoordinate)) does not
	// 		have a coordinate in the input coordinate system.

	// Where are the input and output pixel axes for this coordinate ?
	// DirectionCoordinates and LinearCoordinates (holding two axes) are done in one pass.
	Vector<int> outPixelAxes = outCoords.pixelAxes( outCoordinate );
	Vector<int> inPixelAxes = inCoords.pixelAxes( inCoordinate );

	// Ensure that we are doing a 2d regrid.
	if ( (type == Coordinate::DIRECTION) ||
			(type == Coordinate::LINEAR && outPixelAxes.nelements() == 2 && inPixelAxes.nelements() == 2) )
	{

		// We will do two pixel axes in this pass. mark these axes as being done.
		doneOutPixelAxes( outPixelAxes[0] ) = true;
		doneOutPixelAxes( outPixelAxes[1] ) = true;

		const IPosition inShape = inPtr->shape();
		bool shapeIsDifferent = (outShape( outPixelAxes[0] ) != inShape( inPixelAxes(0) )) ||
					(outShape( outPixelAxes[1] ) != inShape( inPixelAxes(1) ));

		// see if we really need to regrid these axes. if the coordinates and shape are the
		// same then there is nothing to do apart from swap in and out pointers.
		const Coordinate& cIn = inCoords.coordinate( inCoordinate );
		const Coordinate& cOut = outCoords.coordinate( outCoordinate );
		
		bool regridIt = (shapeIsDifferent == true || cIn.near( cOut ) == false );
		if (regridIt == false)
			
			// Axes does not need to be regridded. Just copy the pointers so that the same temporary
			// image can be used as input for the next iteration.
			outPtr = inPtr;
			
		else
		{

			// Axes need regridding. Create a temporary image to store the regridded output image; the temporary
			// image will either be used as input for the next iteration or copied to the final output image.
			int maxMemoryInMB = 0;
			outPtr = new TempImage<T>( TiledShape( outShape ), outCoords, maxMemoryInMB );

			regridTwoAxisCoordinate( *outPtr, *inPtr, inImage.units(), inCoords, outCoords, inCoordinate,
						outCoordinate, inPixelAxes, outPixelAxes, pixelAxisMap1, pixelAxisMap2,
						interpolationMethod, replicate );
		
		}

	}
	else
	{

		// display warning:
		// Can only do a 2d regrid here

	}

} // ImageRegrid<T>::regridOneCoordinate


// Regrid a pair of axes. This subroutine will:
//
//	1. scale an image along either of its two axes. scaling can be to a larger or smaller image size
//	2. re-order the axes of the coordinate system.
//
// Iterate over the output image, using bilinear interpolation to select an appropriate
// output pixel value from the corresponding region of the input image.
//
// This algorithm uses the CASA class LatticeIterator to iterate over the image. A lattice
// is just a n-dimensional data structure (i.e. ra, dec and spectral), which may contain an image. We
// use two difference LatticeIterators here: the first (outIter) is needed to iterate over all
// values of the axes that are NOT being regridded (i.e. if we are regridding ra and dec then
// we iterate over every possible value of the spectral axis, and regrid the 2-d image each time).
//
// The second latticeIterator (outCursorIter) iterates over just the two axes being regridded,
// and this iterator can probably be removed if one is careful to get the syntax right. I think it has
// been included simply to improve performance. The code also has i,j loops to iterate over the pixels
// of the 2-d image, so there really is no need for the second LatticeIterator to be there. I've left
// it in for now because I'm not sure about the correct syntax if it were removed.
//
// The i and j loops that iterate over the output pixels use a class called an ArrayAccessor. This
// simply handles the increments to the row and column pointers, so the code does not need to know
// how the 2-d array is stored in memory. To remove the second LatticeIterator we would need to
// update the ArrayAccessor to handle an n-dimensional (rather that the current 2-d) array.
//
// The output image is updated directly with the result of the interpolation.
//
// outLattice:		the n-dimensional output image
// inLattice:		the n-dimensional input image
// imageUnit:		the flux density units of the image (rescaling happens for units "JY/PIXEL")
// inCoords:		the input coordinate system
// outCoords:		the output coordinate system
// outCoordinate:	the output coordinate ID
// inPixelAxes:		2-d vector with the pixel axis IDs of the input image
// outPixelAxes:	2-d vector with the pixel axis IDs of the output image
// pixelAxisMap1:	says where pixel axis i in the output image is in the input image
// pixelAxisMap2:	says where pixel axis i in the input image is in the output image
// replicate:		true = input and output axes are aligned, false = let CASA do reprojection for us.
//
// Some calculated values are:
//
// xOutAxis:		is the first direction axis in the output image (associated with i)
// yOutAxis:		is the second direction axis in the output image (associated with j)
// xInCorrAxis:		is the corresponding axis to xOutAxis in the input image
// yInCorrAxis:		is the corresponding axis to yOutAxis in the input image
// xInAxis:		is the first direction axis in the input image
// yInAxis:		is the second direction axis in the input image
// xOutCorrAxis:	is the corresponding axis to xInAxis in the output image
// yOutCorrAxis:	is the corresponding axis to yInAxis in the output image

// Example:
//   Regrid ra/dec axes with input order ra/dec/freq and output order freq/dec/ra
//
//   input  image shape = [20, 30, 40] (ra/dec/freq)
//   output image shape = [40, 90, 60] (freq/dec/ra) - we are making ra/dec shape 3x input
//
//   outPixelAxes = [2,1] = [lon,lat]
//   The cursor matrix is of shape [nrow,ncol] = [90,60]
//       xOutAxis = 1   (dec)
//       yOutAxis = 2   (ra)
//    xInCorrAxis = 1   (dec)
//    yInCorrAxis = 0   (ra)
//        xInAxis = 0   (ra)
//        yInAxis = 1   (dec)
//   xOutCorrAxis = 2   (ra)
//   yOutCorrAxis = 1   (dec)

template<class T>
void ImageRegrid<T>::regridTwoAxisCoordinate(	MaskedLattice<T>& outLattice,
						const MaskedLattice<T>& inLattice,
						const Unit& imageUnit,
						const CoordinateSystem& inCoords,
						const CoordinateSystem& outCoords,
						int inCoordinate, int outCoordinate,
						const Vector<int> inPixelAxes,
						const Vector<int> outPixelAxes,
						const Vector<int> pixelAxisMap1,
						const Vector<int> pixelAxisMap2,
						typename Interpolate2D::Method interpolationMethod,
						bool replicate )
{

	// throw error:
	// condition: (inPixelAxes.nelements() != 2 || outPixelAxes.nelements() != 2)
	// message: need 2 axes for both input and output 

	const IPosition inShape = inLattice.shape(), outShape = outLattice.shape();
	const unsigned int nDim = inLattice.ndim();

	// Get the x and y input/output axes, and also where they can be found in the output/input axis list.
	const unsigned int xOutAxis = min( outPixelAxes(0), outPixelAxes(1) );
	const unsigned int yOutAxis = max( outPixelAxes(0), outPixelAxes(1) );
	unsigned int xInCorrAxis = pixelAxisMap1[xOutAxis];
	unsigned int yInCorrAxis = pixelAxisMap1[yOutAxis];
	const unsigned int xInAxis = min( inPixelAxes(0), inPixelAxes(1) );
	const unsigned int yInAxis = max( inPixelAxes(0), inPixelAxes(1) );
	unsigned int xOutCorrAxis = pixelAxisMap2[xInAxis];
	unsigned int yOutCorrAxis = pixelAxisMap2[yInAxis];

	// These tell us which chunk of input data we need to service each iteration through the output image
	IPosition inChunkBlc( nDim );
	IPosition inChunkTrc( nDim );

	// Coordinate conversion vectors - i.e. the scaling factors between the input and output axis.
	Vector<double> pixelScale( nDim, 1.0 );
	pixelScale( xInAxis ) = float( outShape( xOutCorrAxis ) ) / float( inShape( xInAxis ) );
	pixelScale( yInAxis ) = float( outShape( yOutCorrAxis ) ) / float( inShape( yInAxis ) );

	// Create an instance of CASA's 2D interpolator.
	Interpolate2D casa2DInterpolator( interpolationMethod );

	// Create blank array to be the same size as the output image. This array will eventually contain a mapping from
	// output image pixel coordinates to input image pixel coordinates. i.e. output image coordinate <i, j> will correspond
	// to input image coordinate <in2DPos(i, j, 0), in2DPos(i, j, 1)>.
	Cube<double> in2DPos( outShape( xOutAxis ), outShape( yOutAxis ), 2 );
	
	// Construct the 2-d mapping from output to input pixel coordinates. If replicate = true then we do this manually, and
	// if replicate = false then we let CASA do reprojection for us by converting from output pixel coordinates to world
	// coordinates, and then back to input pixel coordinates. The former is for just regridding and the latter is for
	// regridding and reprojection.
	IPosition outPosFull( outLattice.ndim(), 0 );
	if (replicate == true)
		make2DCoordinateGrid(	in2DPos, pixelScale, xInAxis, yInAxis, xOutAxis, yOutAxis, xOutCorrAxis, yOutCorrAxis,
					outShape );
	else
		make2DCoordinateGrid(	in2DPos, inCoords, outCoords, inCoordinate, outCoordinate,
					xInAxis, yInAxis, xOutAxis, yOutAxis, inPixelAxes, outPixelAxes, outPosFull, outShape );

	// Find scale factor for Jy/pixel images. Our final regridded pixel values need to be corrected using
	// this scale factor to insure that the total integrated flux remains the same.
	double scale = findScaleFactor( imageUnit, inCoords, outCoords, inCoordinate, outCoordinate );

	// Construct a vector that gives a cursor shape. This is the portion of the image that is regridded in one iteration.
	// This code specifies which axes will be regridded in this pass. The non-regrid axes are set to 1, which means that
	// there will be a 2-d regridding for EACH separate value of the non-grid axes.
	IPosition niceShape( outShape.nelements ) = 1;
	niceShape( xOutAxis ) = outShape( xOutAxis );
	niceShape( yOutAxis ) = outShape( yOutAxis );

	// CASA uses LatticeStepper and LatticeIterator classes to move through an image in a multi-dimensional coordinate system.
	// We need to construct them here. These classes simply divide up 'outShape' into chunks of size 'niceShape', and then
	// iterate over all the chunks.
	LatticeStepper outStepper( outShape, niceShape, LatticeStepper::RESIZE );
	LatticeIterator<T> outIter( outLattice, outStepper );

	// Iterate through output image one chunk at a time. We are iterating through all possible values on the axes that are
	// NOT being regridded. Each 2-d plane from the regrid axes will be handled in one iteration. i.e. if our image has axes
	// x, y and z, and we are regridding the x and y axis, then we will perform a separate regridding on the xy-plane
	// (i.e. in one iteration) for every possible value of z.
	for ( outIter.reset(); !outIter.atEnd(); outIter++ )
	{

		// Get the bottom-left and top-right coordinates of this chunk of the input image.
		// For the non-regrid axes, the input and output shapes, and hence positions, are the same.
		// pixelAxisMap2(i) says where pixel axis i in the input image is in the output image.
		for ( unsigned int k = 0; k < nDim; k++ )
		{
			inChunkBlc( k ) = outIter.position()[ pixelAxisMap2[k] ];
			inChunkTrc( k ) = outIter.endPosition()[ pixelAxisMap2[k] ];
		}

		// Now overwrite the blc/trc for the regrid axes. We are using the WHOLE 2-d image, and not iterating over it
		// in chunks.
		inChunkBlc( xInAxis ) = 0;
		inChunkBlc( yInAxis ) = 0;
		inChunkTrc( xInAxis ) = inShape( xInAxis ) - 1;
		inChunkTrc( yInAxis ) = inShape( yInAxis ) - 1;
			
		// Calculate the size of input image chunk, and extract it into an array.
		IPosition inChunkShape = inChunkTrc - inChunkBlc + 1;
		Array<T> inDataChunk = inLattice.getSlice( inChunkBlc, inChunkShape );

		// Iterate through the output cursor by Matrices, each holding a Direction plane.
		// This gets us just a few percent speed up over iterating through pixel by pixel.
		ArrayLattice<T> outCursor( outIter.rwCursor() );    // reference copy
	
		// throw error if DirectionCoordinate plane is degenerate (inChunkShape(xInAxis) == 1 && inChunkShape(yInAxis) == 1).
		// throw error if DirectionCoordinate plane has 1 degenerate axis (inChunkShape(xInAxis) == 1 || inChunkShape(yInAxis) == 1).

		// Get the size of the current output image chunk.
		IPosition& outCursorShape = outIter.CursorShape();

		// Create the bottom-left-corner coordinate (0,..,0) and the top-right-corner coordinate
		// (inChunkShape - 1, .., inChunkShape - 1).
		IPosition inChunkBlc2D( nDim, 0 );
		IPosition inChunkTrc2D( nDim ) = inChunkShape - 1;

		// Everything up to this point works with n-dimensions. However, images only use two axes,
		// so we now construct some objects that use just x and y axes.
		// Create a 2-d object which gives the size (shape) of the chunk being interpolated.
		IPosition inChunk2DShape( 2, inChunkShape[xInAxis], inChunkShape[yInAxis] );

		// We now need another LatticeStepper and LatticeIterator, this time to iterate over the 2-d array. This time we
		// pass in the x and y output axes (outCursorAxes) so the stepper knows to only iterate over these axes.
		IPosition axisPath;
		IPosition outCursorAxes( 2, xOutAxis, yOutAxis );
		IPosition outCursorIterShape( 2, outCursorShape( xOutAxis ), outCursorShape( yOutAxis ) );
			
		LatticeStepper outCursorStepper( outCursor.shape(), outCursorIterShape, outCursorAxes, axisPath );
		LatticeIterator<T> outCursorIter( outCursor, outCursorStepper );

		for ( outCursorIter.reset(); !outCursorIter.atEnd(); outCursorIter++ )
		{
    
			// outIter.position() is the location of the current cursor (tile) of data within the full lattice
			// outCursorIter.position is the location of the BLC of the current matrix within the current cursor (tile)
			// of data. the new value outPos is the location of the BLC of the current matrix within the full lattice.
			IPosition outPos = outIter.position() + outCursorIter.position();
    
			// Fish out the 2D piece of the inChunk relevant to this plane of the cursor
			for ( unsigned int k = 0; k < nDim; k++ )
				if ( k != xInAxis && k != yInAxis )
				{
					inChunkBlc2D[k] = outPos[pixelAxisMap2[k]] - inChunkBlc[k];
					inChunkTrc2D[k] = inChunkBlc2D[k];
				}

			const Matrix<T>& inDataChunk2D = inDataChunk( inChunkBlc2D, inChunkTrc2D ).reform( inChunk2DShape );
    
			// Define array accessors that loop over axis 0 (x-axis). These are effectively just pointers to an array,
			// but by specifying an axis we can increment these pointers (i.e. gridRowPointer++) and have them
			// automatically move by a whole column/row rather than just move one pixel in whatever direction the
			// 2-D array is flattened.
			ArrayAccessor<T, Axis<0>> gridPointer;

			// Define array accessor that loops over axis 1 (y-axis). We also supply the 2-d data from the output
			// chunk in CASA Matrix format. This allows the ArrayAccessor to increment the row pointer.
			ArrayAccessor<T, Axis<1>> gridRowPointer( outCursorIter.rwMatrixCursor() );
				
			// Get the number of rows and columns.
			unsigned int nRow = outCursorIter.matrixCursor().nrow();
			unsigned int nCol = outCursorIter.matrixCursor().ncolumn();

			// the x image loop.
			for ( unsigned int j = 0; j < nRow; j++ )
			{

				// initialise array accessors to point to the start of the row.
				gridPointer = gridRowPointer;

				// the y image loop.
				for ( unsigned int i = 0; i < nCol; i++ )
				{
						
					// Initialise output pixel value to 0.
					*gridPointer = 0.0;
	  
					// Now do the interpolation. in2DPos(i,j) is the absolute input pixel coordinate in the input
					// lattice for the current output pixel.
					Vector<double> pix2DPos2(2);
					pix2DPos2[0] = in2DPos( outPos[xOutAxis] + i, outPos[yOutAxis] + j, 0 ) - inChunkBlc[xInAxis];
					pix2DPos2[1] = in2DPos( outPos[xOutAxis] + i, outPos[yOutAxis] + j, 1 ) - inChunkBlc[yInAxis];
							
					// Perform a bilinear interpolation about the pixels at position 'pix2DPos2' in the input image
					// 'inDataChunk2D' to get an output pixel value 'result'.
					T result(0);
					bool interpOK = casa2DInterpolator.interp( result, pix2DPos2, inDataChunk2D );
					if (interpOK == true)
						*gridPointer = scale * result;

					// move 1 pixel in the x-direction.
					gridPointer++;

				}

				// move 1 pixel in the y-direction.
				gridRowPointer++;

			}
			
		}
	}

} // ImageRegrid<T>::regridTwoAxisCoordinate


// Perform validation on a pair of input and output axes.
//
// outPixelAxes:	a list of pixel axes to regrid
// inShape:		the size of each axis for the n-dimensional input image
// outShape:		the size of each axis for the n-dimensional output image
// pixelAxisMap1:	says where pixel axis i in the output image is in the input image
// outCoords:		the output image coordinate system

template<class T>
void ImageRegrid<T>::checkAxes(	IPosition& outPixelAxes,
                               	const IPosition& inShape,
                               	const IPosition& outShape,
                               	const Vector<int>& pixelAxisMap1,
                               	const CoordinateSystem& outCoords )
{

	// throw error if input shape is illegal (inShape.nelements() == 0).

	int n1 = outPixelAxes.nelements();
	const int nOut = outShape.nelements();

	// throw error if # pixel axes is more than dimensions (n1 > nOut).

	// fill in all axes if null pixelAxes given
	if (n1 == 0)
	{
		outPixelAxes = IPosition::makeAxisPath( nOut );
		n1 = outPixelAxes.nelements();
	}

	// Check for Stokes and discard
	int outCoordinate, outAxisInCoordinate;
	int j = 0;
	for ( int i = 0; i < n1; i++ )
	{

		// find pixel axis in output coordinates if not yet done
      		outCoords.findPixelAxis( outCoordinate, outAxisInCoordinate, outPixelAxes( i ) );

		// throw error if pixel axis has been removed from the output coordinate system (outCoordinate == -1 || outAxisInCoordinate == -1).

		// find out the coordinate type and don't allow Stokes
		Coordinate::Type type = outCoords.type( outCoordinate );

		if (type != Coordinate::STOKES)
		{
			bool ok = true;
			if (outShape( outPixelAxes( i ) ) == 1)
			{

				if (type != Coordinate::DIRECTION)
		      	        	ok = false;

        		 } 

         		if (ok == true)
			{
            			outPixelAxes( j ) = outPixelAxes( i );
            			j++;
         		}
      		}
	}

	outPixelAxes.resize( j, true );
	n1 = outPixelAxes.nelements();

	// check for range
	Vector<bool> found( nOut, false );
	for ( int i = 0; i < n1; i++ )
	{

		// throw error if pixel axes are out of range (outPixelAxes( i ) < 0 || outPixelAxes( i ) >= nOut).

		// throw error if specified pixel axes are not unique (found( outPixelAxes( i ) ) == true ).

		found( outPixelAxes( i ) ) = true;

	}

	// check non-regridded axis shapes are ok
	for ( int i = 0; i < nOut; i++ )
	{
		bool foundIt = false;
		for ( int j = 0; j < n1; j++ )
		{
			if (outPixelAxes( j ) == i)
			{
				foundIt = true;
				break;
			}
 		}

		// pixelAxisMap1(i) says where pixel axis i in the output image
		// is in the input image.

		// throw an error if input and output axis have different shapes (!foundIt && outShape( i ) != inShape( pixelAxisMap1[i] )).

	}

} // ImageRegrid<T>::checkAxes


// returns two vectors which map the axes of one coordinate system to another. pixelAxisMap1 gives
//	the positions of each output pixel axis in the input axes, and pixelAxisMap2 gives the positions
//	of each input pixel axis in the output axes.
//
// this function maps the array indexes only (i.e. it is NOT a transformation) - the same axes must
// 	exist in both co-ordinate systems or the return value will be -1.
//
// nDim:		number of dimensions in the input coordinate system
// pixelAxisMap1:	[return] gives position of each output pixel axis i in the input coordinate system
// pixelAxisMap2:	[return] gives position of each input pixel axis i in the output coordinate system
// inCoords:		input coordinate system
// outCoords:		output coordinate system

template<class T>
void ImageRegrid<T>::findMaps(	unsigned int nDim, 
				Vector<int>& pixelAxisMap1,
				Vector<int>& pixelAxisMap2,
				const CoordinateSystem& inCoords,
				const CoordinateSystem& outCoords ) const
{

	// worldAxisMap gives location each input world axis i in output axes.
	// worldAxisTranspose gives location of each output world axis i in input axes.
	Vector<int> worldAxisTranspose, worldAxisMap;
	Vector<bool> worldRefChange;

	// get a map (worldAxisTranspose) between the input and output world axes. returns false if either coordinate system
	// has no valid axes.
	outCoords.worldMap( worldAxisMap, worldAxisTranspose, worldRefChange, inCoords )

	// resize vectors to the number of dimensions in the input coordinate system.
	pixelAxisMap1.resize(nDim);
	pixelAxisMap2.resize(nDim);

	for ( unsigned int paOut = 0; paOut < nDim; paOut++ )
	{

		// switch from pixel axis to world axis, transponse from output to input, and then switch back to pixel axis.
		int waOut = outCoords.pixelAxisToWorldAxis( paOut );
		int waIn = worldAxisTranspose( waOut );
		int paIn = inCoords.worldAxisToPixelAxis( waIn );      

		pixelAxisMap1[paOut] = paIn;
		pixelAxisMap2[paIn] = paOut;

	}

} // ImageRegrid<T>::findMaps


// Builds a 3-d grid (in2DPos), where the first two dimensions are the x and y axis of the output image. The 3rd dimension
// has elements 0 or 1, where 0 corresponds to x and 1 to y. The newly built array has values:
//
//		in2DPos( i, j, 0 ) = x axis scaling factor (output / input) multiplied by i (if axes are not being swapped)
//						or j (if axes are being swapped)
//		in2DPos( i, j, 1 ) = y axis scaling factor (output / input) multiplied by j (if axes are not being swapped)
//						or i (if axes are being swapped)
//
// This function therefore returns an array that can be used to scale directly from output image pixel coordinates to
// input image pixel coordinates. For example, our image pixel coordinates <i, j> map directly to input image pixel
// coordinates <in2DPos(i, j, 0), in2DPos(i, j, 1)>.
//
// This overloaded version of make2DCoordinateGrid() will regrid the image but will NOT reproject the coordinates onto a
// different coordinate system. It can only be used if the input and output coordinate system is completely aligned. In other
// words, this function is intended for just a 2-d rescaling of the input image.
//
// in2DPos:		the return grid
// pixelScale:		an n-dimensional vector giving the scaling between the output and input axes.
// xInAxis:		the x direction axis in the input image
// yInAxis:		the y direction axis in the input image
// xOutAxis:		the x direction axis in the output image
// yOutAxis:		the y direction axis in the output image
// xOutCorrAxis:	the output axis corresponding to the input image x axis
// yOutCorrAxis:	the output axis corresponding to the input image y axis
// outCursorShape:	contains the dimensions of the image

template<class T>
void ImageRegrid<T>::make2DCoordinateGrid(	Cube<double>& in2DPos,
						const Vector<double>& pixelScale,
						unsigned int xInAxis,
						unsigned int yInAxis,
						unsigned int xOutAxis,
						unsigned int yOutAxis,
						unsigned int xOutCorrAxis,
						unsigned int yOutCorrAxis,
						const IPosition& outCursorShape )
{

	switch (xOutAxis)
	{
		// x output axis corresponds to x input axis.
		case xOutCorrAxis:    
			for ( unsigned int j = 0; j < outCursorShape( yOutAxis ); j++ )
				for ( unsigned int i = 0; i < outCursorShape( xOutAxis ); i++ )
				{
					in2DPos( i, j, 0 ) = ((double(i) + 0.5) / pixelScale( xInAxis )) - 0.5;
					in2DPos( i, j, 1 ) = ((double(j) + 0.5) / pixelScale( yInAxis )) - 0.5;
				}
			break;
			
		// x output axis corresponds to y input axis (axes are swapped).
		case yOutCorrAxis:
			for ( unsigned int j = 0; j < outCursorShape( yOutAxis ); j++ )
				for ( unsigned int i = 0; i < outCursorShape( xOutAxis ); i++ )
				{
					in2DPos( i, j, 0 ) = ((double(j) + 0.5) / pixelScale( xInAxis )) - 0.5;
					in2DPos( i, j, 1 ) = ((double(i) + 0.5) / pixelScale( yInAxis )) - 0.5;
				}
			break;
	}
   
} // ImageRegrid<T>::make2DCoordinateGrid


// This overloaded version of make2DCoordinateGrid also returns an array that can be used to scale directly from output image pixel
// coordinates to input image pixel coordinates. However, in this case we do not construct the array ourselves but use CASA to
// automatically convert the coordinates for us. The output coordinates are converted to world coordinates, and then back into
// pixel coordinates in the input coordinate system. CASA will automatically do UM conversion for us.
//
// The main difference between this overloaded version and the simpler one above is that this code is capable of doing
// reprojection as well as regridding. The output pixel coordinates are converted into world coordinates by CASA and then
// converted into input pixel coordinates. If the input and output coordinate systems are not quite aligned then CASA will do a
// SIN reprojection of the coordinates.
//
// in2DPos:		the return grid
// inCoords:		the input image coordinate system
// outCoords:		the output image coordinate system
// inCoordinate:	the input coordinate ID
// outCoordinate:	the output coordinate ID
// xInAxis:		the x direction axis in the input image
// yInAxis:		the y direction axis in the input image
// xOutAxis:		the x direction axis in the output image
// yOutAxis:		the y direction axis in the output image
// inPixelAxes:		the input image pixel axes
// outPixelAxes:	the output image pixel axes
// outPos:		the blc of the image chunk currently being regridded
// outCursorShape:	the size of the output image coordinates (in order to loop over the x and y grid)

template<class T>
void ImageRegrid<T>::make2DCoordinateGrid(	Cube<double>& in2DPos,
						const CoordinateSystem& inCoords,
						const CoordinateSystem& outCoords,
						int inCoordinate,
						int outCoordinate,
						unsigned int xInAxis,
						unsigned int yInAxis,
						unsigned int xOutAxis
						unsigned int yOutAxis,
						const IPosition& inPixelAxes,
						const IPosition& outPixelAxes,
						const IPosition& outPos,
						const IPosition& outCursorShape )
{
	
	// in2DPos says where the output pixel (i,j) is located in the input image
	Vector<double> world(2), inPixel(2), outPixel(2);
  
	// Where in the Direction Coordinates are X and Y ?
	// pixelAxes(0) says where Lon is in DirectionCoordinate
	// pixelAxes(1) says where Lat is in DirectionCoordinate
	// The X axis is always the direction axis that appears first in the image

	unsigned int inXIdx = 0, inYIdx = 1;	// [x,y] = [lon,lat]
	if (inPixelAxes(0) == int( yInAxis ))
		swap( inXIdx, inYIdx );		// [x,y] = [lat,lon]

	unsigned int outXIdx = 0, outYIdx = 1;
	if (outPixelAxes(0) == int(yOutAxis))
		swap( outXIdx, outYIdx );

	// Are we dealing with a DirectionCoordinate or LinearCoordinate ?
	bool isDirection = (inCoords.type( inCoordinate ) == Coordinate::DIRECTION && 
				outCoords.type( outCoordinate ) == Coordinate::DIRECTION);
	DirectionCoordinate inDir, outDir;
	LinearCoordinate inLin, outLin;
	if (isDirection == true)
	{
		
		inDir = inCoords.directionCoordinate( inCoordinate );
		outDir = outCoords.directionCoordinate( outCoordinate );
		
	}
	else
	{
		
		inLin = inCoords.linearCoordinate( inCoordinate );
		outLin = outCoords.linearCoordinate( outCoordinate );

		// Set units to same thing
		const Vector<String>& units = inLin.worldAxisUnits().copy();
		
		// throw error:
		// condition: outLin.setWorldAxisUnits( units ) == false
		// message: Failed to set output and input LinearCoordinate axis units the same
		
	)

	// Compute all coordinates (very expensive).
	for ( unsigned int j = 0; j < outCursorShape( yOutAxis ); j += 1 )
		for ( unsigned int i = 0; i < outCursorShape( xOutAxis ); i += 1 )
		{
			
			outPixel( outXIdx ) = i + outPos[xOutAxis];
			outPixel( outYIdx ) = j + outPos[yOutAxis];

			// Do coordinate conversions (outpixel to world to inpixel) for the axes of interest
			bool ok = false;
			if (isDirection == true)
			{
				ok = outDir.toWorld( world, outPixel );
				if (ok == true)
					ok = inDir.toPixel( inPixel, world );
			}
			else
			{
				ok = outLin.toWorld( world, outPixel );
				if (ok == true)
					ok = inLin.toPixel( inPixel, world );
			}

			if (ok1 == true)
			{

				// This gives the 2D input pixel coordinate (relative to the start of the full Lattice)
				// to find the interpolated result at.  (,,0) pertains to inX and (,,1) to inY
				in2DPos( i, j, 0 ) = inPixel( inXIdx );
				in2DPos (i, j, 1 ) = inPixel( inYIdx );
				
			}
		}
	
} // ImageRegrid<T>::make2DCoordinateGrid


// Returns the scaling factor (output / input) between two axes. input image units must be JY/PIXEL. For
// direction coordinates the units of both axes are converted into degrees.
//
// The result of this function is used to scale the output pixel values so that the total integrated
// flux remains constant when scaling an image along the x and y axes.
//
// units:		the flux density units of the image
// inCoords:		the input image coordinate system
// outCoords:		the output image coordinate system
// inCoordinate:	the input coordinate integer ID
// outCoordinate:	the output coordinate integer ID

template<class T>
double ImageRegrid<T>::findScaleFactor(	const Unit& units, 
                                       	const CoordinateSystem& inCoords, 
					const CoordinateSystem& outCoords,
                                       	int inCoordinate,
					int outCoordinate ) const
{

	double fac = 1.0;
	String t = units.getName();
	t.upcase();
	if (t == String( "JY/PIXEL" ))
	{

		// set units to the same thing
      		if (inCoords.type( inCoordinate ) == Coordinate::DIRECTION)
		{

         		DirectionCoordinate inDir = inCoords.directionCoordinate( inCoordinate );
         		DirectionCoordinate outDir = outCoords.directionCoordinate( outCoordinate );

        		Vector<String> units( 2 );
         		units.set( "deg" );

			inDir.setWorldAxisUnits( units );
			outDir.setWorldAxisUnits( units );

			const Vector<double>& incIn = inDir.increment();
			const Vector<double>& incOut = outDir.increment();

			fac = abs( incOut( 0 ) * incOut( 1 ) / incIn( 0 ) / incIn( 1 ) );

		}
		else if (inCoords.type( inCoordinate ) == Coordinate::LINEAR)
		{

			// display warning:
			// Can only do a 2d regrid here

		}

	}

	return fac;

} // ImageRegrid<T>::findScaleFactor


} //# NAMESPACE CASACORE - END


#endif
