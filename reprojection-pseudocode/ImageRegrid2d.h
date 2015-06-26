//# ImageRegrid.h: Regrid Images

#ifndef IMAGES_IMAGEREGRID_H
#define IMAGES_IMAGEREGRID_H

#include <casacore/casa/aips.h>
#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Cube.h>
#include <casacore/measures/Measures/MDirection.h>
#include <casacore/measures/Measures/MFrequency.h>
#include <casacore/scimath/Mathematics/Interpolate2D.h>
#include <set>

namespace casacore { //# NAMESPACE CASACORE - BEGIN

template<class T> class MaskedLattice;
template<class T> class ImageInterface;
template<class T> class Lattice;
template<class T> class LatticeIterator;
template<class T> class Vector;

class CoordinateSystem;
class DirectionCoordinate;
class Coordinate;
class ObsInfo;
class IPosition;
class Unit;

// <summary>This regrids one image to match the coordinate system of another</summary>

// <use visibility=export>

// <reviewed reviewer="" date="yyyy/mm/dd" tests="" demos="">
// </reviewed>

// <prerequisite>
//   <li> <linkto class="ImageInterface">ImageInterface</linkto>
//   <li> <linkto class="CoordinateSystem">CoordinateSystem</linkto>
//   <li> <linkto class="Interpolate2D">Interpolate2D</linkto>
//   <li> <linkto class="InterpolateArray1D">InterpolateArray1D</linkto>
// </prerequisite>
//
// <etymology>
//  Regrids, or resamples, images.  
// </etymology>
//
// <synopsis>
//  This class enables you to regrid one image to the coordinate
//  system of another.    You can regrid any or all of the
//  axes in the image.  A range of interpolation schemes are available.
//
//  It will cope with coordinate systems being in different orders
//  (coordinate, world axes, pixel axes).  The basic approach is to
//  make a mapping from the input to the output coordinate systems,
//  but the output CoordinateSystem order is preserved in the output
//  image.
//
//  Any DirectionCoordinate or LinearCoordinate holding exactly two axes
//  is regridded in one pass with a 2-D interpolation scheme.
//  All other axes are regridded in separate passes with a 1D interpolation 
//  scheme.    This means that a LinearCoordinate holding say 3 axes
//  where some of them are coupled will not be correctly regridded.
//  StokesCoordinates cannot be  regridded.
//
//  Multiple passes are made through the data, and the output of 
//  each pass is the input of the next pass.  The intermediate 
//  images are stored as TempImages which may be in memory or 
//  on disk, depending on their size.
//
//  It can also simply insert this image into that one via
//  an integer shift.
// </synopsis>
//
// <example>
// 
// <srcblock>
// </srcblock>
// </example>
//
// <motivation> 
// A common image analysis need is to regrid images, e.g. to compare
// images from different telescopes.
// </motivation>
//
// <thrown>
// <li> AipsError 
// </thrown>
//
// <todo asof="1999/04/20">
// </todo>

template <class T> class ImageRegrid2d
{

public:

	// Default constructor
	ImageRegrid2d();

	// destructor
	~ImageRegrid();

	// Regrid inImage onto the grid specified by outImage.
	// If outImage has a writable mask, it will be updated in that output pixels at which the regridding failed will
	// be masked bad (False) and the pixel value set to zero. Otherwise the output mask is not changed.
	// Specify which pixel axes of outImage are to be regridded.  The coordinate and axis order of outImage
	// is preserved, regardless of where the relevant coordinates are in inImage.

	void regrid(	ImageInterface<T>& outImage, 
			typename Interpolate2D::Method interpolationMethod,
		 	const IPosition& whichOutPixelAxes,
			const ImageInterface<T>& inImage,
			bool replicate );

 private:
	
	LogIO _os(LogOrigin("ImageRegrid", WHERE));

	// Regrid one Coordinate
	void regridOneCoordinate(	Vector<bool>& doneOutPixelAxes,
					MaskedLattice<T>* &inPtr,   
					MaskedLattice<T>* &outPtr,  
					CoordinateSystem& outCoords,
					const CoordinateSystem& inCoords,
					int outPixelAxis,
					const ImageInterface<T>& inImage,
					const IPosition& outShape,
					typename Interpolate2D::Method interpolationMethod,
					bool replicate );

	// Regrid  DirectionCoordinate or 2-axis LinearCoordinate
	void regridTwoAxisCoordinate(	MaskedLattice<T>& outLattice,
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
					bool replicate );
 
	// Check shape and axes.  Exception if no good.  If pixelAxes
	// of length 0, set to all axes according to shape
	void checkAxes(	IPosition& outPixelAxes,
			const IPosition& inShape,
			const IPosition& outShape,
			const Vector<int>& pixelAxisMap,
			const CoordinateSystem& outCoords );

	// Find maps between coordinate systems
	void findMaps (	unsigned int nDim, 
				Vector<int>& pixelAxisMap1,
				Vector<int>& pixelAxisMap2,
				const CoordinateSystem& inCoords,
				const CoordinateSystem& outCoords ) const;

	// Make replication coordinate grid for this cursor
	void make2DCoordinateGrid(	Cube<double>& in2DPos,
					const Vector<double>& pixelScale, 
					unsigned int xInAxis,
					unsigned int yInAxis,
					unsigned int xOutAxis,
					unsigned int yOutAxis,
					unsigned int xOutCorrAxis,
					unsigned int yOutCorrAxis,
					const IPosition& outCursorShape );

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

	// Find scale factor to conserve flux 
	double findScaleFactor(	const Unit& units, 
  				const CoordinateSystem& inCoords, 
				const CoordinateSystem& outCoords, 
				int inCoordinate, int outCoordinate ) const;

};


} //# NAMESPACE CASACORE - END

#ifndef CASACORE_NO_AUTO_TEMPLATES
#include <casacore/images/Images/ImageRegrid.tcc>
#endif //# CASACORE_NO_AUTO_TEMPLATES
#endif

