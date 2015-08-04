//
//	reprojection.c
//
//	Chris Skipper
//	07/07/2015
//
//	Performs a regridding and a reprojection on an input image. The input and output coordinate systems are
//	based upon the FITS standard (see below), and must be supplied as parameters in the 'reprojection-params' file.
//
//	We use the definition of a coordinate system used in FITS files. This definition is based upon a single
//	reference pixel within each image:
//
//	crVAL1, crVAL2:		the RA and Dec of the reference pixel (α_r, δ_r).
// 	crPIX1, crPIX2:		the x and y pixel coordinates of the reference pixel.
//	cd1_1, .., cd2_2:	a 2x2 matrix (in degrees/pixel) which transforms the pixel coordinates (given here as offsets
//				from crPIX1, crPIX2) into intermediate coordinates, such that:
//
//					( intermediate_1 ) =	( cd1_1	cd1_2 )	 ( pixel_1 )
//					( intermediate_2 )	( cd2_1	cd2_2 )  ( pixel_2 )
//
//				i.e., if pixel coordinates are aligned with RA and dec, with 1 arcsec/pixel, then the matrix
//				would be:
//
//					CD =	( -1 / 3600	0        )
//						( 0		1 / 3600 )
//
//				intermediate coordinates give the offset from the reference pixel in the RA and dec directions,
//				but do so as angular sizes (in degrees). to convert to RA and dec offsets we need to skew the
//				image to take account of the fact that lines of RA get closer together in angular size as
//				the images gets nearer to +/- 90 degrees declination. to do this we use:
//
//					world_1 (i.e. RA, α) = intermediate_1 / cos( intermediate_2 )
//					world_2 (i.e. DEC, δ) = intermediate_2
//
//				note that the pixel and world coordinates in pixel_1, pixel_2, world_1 and world_2 are offsets
//				from (crPIX1, crPIX2) and (crVAL1, crVal2) respectively.
//
//				FITS files will normally specify the type of projection that was used in their construction in
//				their header information, but here I am restricting my code to handling one type of
//				projection only: lines of declination are parallel, and the angular separation
//				between lines of right ascension scales with declination as cos(dec).
//
//	This software works with bitmap (BMP) files, which makes it easier to test. BMPs should be 8-bit/pixel, and will be
//	converted into greyscale if they are not already so. I have tested saving bitmaps in GIMP 2.8, and these will work if
//	the image mode is set to Greyscale (Image -> Mode -> Greyscale), but I have not tested images created in Photoshop.
//
//	Note that the CASA version of regridding and reprojection uses the WCSLIB library, which was originally developed
//	to handle coordinate systems in FITS files, and will perform a much more rigorous spherical projection of the pixel
//	coordinates than the approach adopted here. For more information on how WCSLIB works, see:
//
//		Greisen and Calabretta 2002 A&A 395 1061-1075 (Paper I)
//		Calabretta and Greisen 2002 A&A 395 1077-1122 (Paper II)
//
//	Parameters:
//
//		1	input filename (bitmap)
//		2	output filename (bitmap)
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>

//
//	ENUMERATED TYPES
//

typedef enum
{
	EPOCH_J2000,
	EPOCH_B1950,
	EPOCH_GALACTIC
} Epoch;

//
//	STRUCTURES
//

// vector with floats. Can be used either as a 2 or 3 element vector.
struct VectorF
{
	double x;
	double y;
	double z;
};
typedef struct VectorF VectorF;

// vector with integers.
struct VectorI
{
	int x;
	int y;
};
typedef struct VectorI VectorI;

// 2x2 matrix.
struct Matrix2x2
{
	double a11;
	double a12;
	double a21;
	double a22;
};
typedef struct Matrix2x2 Matrix2x2;

// 3x3 matrix.
struct Matrix3x3
{
	double a11;
	double a12;
	double a13;
	double a21;
	double a22;
	double a23;
	double a31;
	double a32;
	double a33;
};
typedef struct Matrix3x3 Matrix3x3;

// stores a coordinate system - reference pixel coordinates and world coordinates, and transformation matrix CD.
struct CoordinateSystem
{
	
	// the reference pixel.
	VectorF crVAL;
	VectorI crPIX;
	
	// the linear conversion between pixel coordinates and RA and DEC offsets.
	Matrix2x2 cd;
	Matrix2x2 inv_cd;
	
	// the rotation matrices to convert between coordinates near the origin (RA 0, DEC 0) to the required RA and DEC.
	Matrix3x3 toWorld;
	Matrix3x3 toPixel;
	
	// an Epoch enumerated type. either J2000, B1950 or GALACTIC.
	Epoch epoch;
	
};
typedef struct CoordinateSystem CoordinateSystem;

//
//	CONSTANTS
//

const double PI = 3.14159265359;
	
// define a maximum number of interpolation points for each output pixel. this is the maximum along each axis, so the actual
// number of interpolation points is n*n per pixel.
const int MAX_INTERPOLATION_POINTS = 10;

// default pixel value - usually black (0) or white (255).
const int DEFAULT_PIXEL_VALUE = 0;
	
// coordinates of the Galactic coordinate system north pole in the J2000 coordinate system.
const double NP_RA_GAL_IN_J2000 = 192.859496;
const double NP_DEC_GAL_IN_J2000 = 27.128353;
const double NP_RA_OFFSET_GAL_IN_J2000 = 302.932069;
	
// coordinates of the J2000 coordinate system north pole in the galactic coordinate system.
const double NP_RA_J2000_IN_GAL = 122.932000;
const double NP_DEC_J2000_IN_GAL = 27.128431;
const double NP_RA_OFFSET_J2000_IN_GAL = 12.860114;
	
// coordinates of the Galactic coordinate system north pole in the B1950 coordinate system.
const double NP_RA_GAL_IN_B1950 = 192.250000;
const double NP_DEC_GAL_IN_B1950 = 27.400000;
const double NP_RA_OFFSET_GAL_IN_B1950 = 303.000000;
	
// coordinates of the B1950 coordinate system north pole in the galactic coordinate system.
const double NP_RA_B1950_IN_GAL = 123.000000;
const double NP_DEC_B1950_IN_GAL = 27.400000;
const double NP_RA_OFFSET_B1950_IN_GAL = 12.250000;
	
// coordinates of the J2000 coordinate system north pole in the B1950 coordinate system.
const double NP_RA_J2000_IN_B1950 = 359.686210;
const double NP_DEC_J2000_IN_B1950 = 89.721785;
const double NP_RA_OFFSET_J2000_IN_B1950 = 0.327475;
	
// coordinates of the B1950 coordinate system north pole in the J2000 coordinate system.
const double NP_RA_B1950_IN_J2000 = 180.315843;
const double NP_DEC_B1950_IN_J2000 = 89.72174782;
const double NP_RA_OFFSET_B1950_IN_J2000 = 179.697628;

// the input parameters from file reprojection-params. these define the input and output
// coordinate systems using FITS convention.
const char IN_CRVAL1[] = "in_crval1:";
const char IN_CRVAL2[] = "in_crval2:";
const char IN_CRPIX1[] = "in_crpix1:";
const char IN_CRPIX2[] = "in_crpix2:";
const char IN_CD1_1[] = "in_cd1_1:";
const char IN_CD1_2[] = "in_cd1_2:";
const char IN_CD2_1[] = "in_cd2_1:";
const char IN_CD2_2[] = "in_cd2_2:";
const char IN_EPOCH[] = "in_epoch:";
const char OUT_SIZE_1[] = "out_size_1:";
const char OUT_SIZE_2[] = "out_size_2:";
const char OUT_CRVAL1[] = "out_crval1:";
const char OUT_CRVAL2[] = "out_crval2:";
const char OUT_CRPIX1[] = "out_crpix1:";
const char OUT_CRPIX2[] = "out_crpix2:";
const char OUT_CD1_1[] = "out_cd1_1:";
const char OUT_CD1_2[] = "out_cd1_2:";
const char OUT_CD2_1[] = "out_cd2_1:";
const char OUT_CD2_2[] = "out_cd2_2:";
const char OUT_EPOCH[] = "out_epoch:";

// bitmap file header positions.
const int BIT_CONST = 0x00;
const int MAP_CONST = 0x01;
const int IMAGE_SIZE = 0x02;
const int RESERVED = 0x06;
const int FILE_HEADER_SIZE = 0x0A;
const int BITMAP_INFO_HEADER = 0x0E;
const int IMAGE_WIDTH = 0x12;
const int IMAGE_HEIGHT = 0x16;
const int COLOUR_PLANES = 0x1A;
const int BIT_COUNT = 0x1C;
const int COMPRESSION_TYPE = 0x1E;
const int COLOURS_USED = 0x2E;
const int SIGNIFICANT_COLOURS = 0x32;

//
//	GLOBAL VARIABLES
//

// input image and coordinate system.
VectorI _inSize = { .x = -1, .y = -1 };
CoordinateSystem _inCoordinateSystem;
float _inFluxScale = 0;

// output image and coordinate system.
VectorI _outSize = { .x = -1, .y = -1 };
CoordinateSystem _outCoordinateSystem;
float _outFluxScale = 0;
	
// input and output images.
unsigned char * _inputImage;
unsigned char * _outputImage;

// rotation matrix for epoch conversion. this is built once at the start.
Matrix3x3 _epochConversion = { .a11 = 1, .a12 = 0, .a13 = 0, .a21 = 0, .a22 = 1, .a23 = 0, .a31 = 0, .a32 = 0, .a33 = 1 };

//
//	GENERAL FUNCTIONS
//

//
//	minF(), minI()
//
//	CJS: 08/07/2015
//
//	Find the minimum of two values. Double and Int types implemented.
//

double minF( double pA, double pB )
{
	
	return (pA < pB) ? pA : pB;
	
} // minF

int minI( int pA, int pB )
{
	
	return (pA < pB) ? pA : pB;
	
} // minI

//
//	maxF(), maxI()
//
//	CJS: 08/07/2015
//
//	Find the maximum of two values. Double and Int types implemented.
//

double maxF( double pA, double pB )
{
	
	return (pA > pB) ? pA : pB;
	
} // maxF

int maxI( int pA, int pB )
{
	
	return (pA > pB) ? pA : pB;
	
} // maxI

//
//	toUppercase()
//
//	CJS: 03/08/2015
//
//	Convert a string to uppercase.
//

void toUppercase( char * pChar )
{
	
	for ( char * ptr = pChar; *ptr; ptr++ )
		*ptr = toupper( *ptr );
	
} // toUppercase

//
//	TRIG FUNCTIONS
//

//
//	rad(), deg()
//
//	CJS:	10/07/2015
//
//	convert between degrees and radians.
//

double rad( double pIn )
{
	
	return ( pIn * PI / (double)180 );
	
} // rad

double deg( double pIn )
{
	
	return ( pIn * (double)180 / PI );
	
} // deg

//
//	arctan()
//
//	CJS:	10/07/2015
//
//	calculate the arctangent of top / bottom, and convert to degrees.
//

double arctan( double pValueTop, double pValueBottom )
{
	
	// calculate arctangent of top / bottom.
	double result;
	if (pValueBottom != 0)
		result = deg( atan( pValueTop / pValueBottom ) );
	else
		result = (pValueTop >= 0) ? 90 : -90;
	
	// we should have a result between -90 and +90 deg. if the denominator is negative then add 180 to be within
	// range 90 to 270 deg.
	if (pValueBottom < 0)
		result = result + 180;
	
	return result;
	
} // arctan

//
//	angleRange()
//
//	CJS:	10/07/2015
//
//	Ensure that an angle is within a given range by adding +/-n.360.
//

void angleRange( double * pValue, double pCentre )
{
	
	while (*pValue < (pCentre - 180))
		*pValue = *pValue + 360;
	while (*pValue > (pCentre + 180))
		*pValue = *pValue - 360;
	
} // angleRange

//
//	MATRIX FUNCTIONS
//

//
//	calculateInverseMatrix()
//
//	CJS:	08/07/2015
//
//	calculate the inverse of a 2x2 matrix.
//

Matrix2x2 calculateInverseMatrix( Matrix2x2 pMatrix )
{
	
	// calculate the determinant.
	double determinant = (pMatrix.a11 * pMatrix.a22) - (pMatrix.a12 * pMatrix.a21);
		
	// create an empty matrix.
	Matrix2x2 inverse = { .a11 = 0, .a12 = 0, .a21 = 0, .a22 = 0 };
	
	// the matrix is supposed to be invertible, but we should check anyway.
	if (determinant != 0)
	{
		inverse.a11 = pMatrix.a22 / determinant;
		inverse.a12 = -pMatrix.a12 / determinant;
		inverse.a21 = -pMatrix.a21 / determinant;
		inverse.a22 = pMatrix.a11 / determinant;
	}
	
	// return the inverse matrix.
	return inverse;
	
} // calculateInverseMatrix

//
//	transpose()
//
//	CJS: 28/07/2015
//
//	Construct the transpose of a 3x3 matrix.
//

Matrix3x3 transpose( Matrix3x3 pOldMatrix )
{
	
	Matrix3x3 newMatrix = pOldMatrix;
	
	// copy transposed cells.
	newMatrix.a12 = pOldMatrix.a21;
	newMatrix.a13 = pOldMatrix.a31;
	newMatrix.a21 = pOldMatrix.a12;
	newMatrix.a23 = pOldMatrix.a32;
	newMatrix.a31 = pOldMatrix.a13;
	newMatrix.a32 = pOldMatrix.a23;
	
	// return something.
	return newMatrix;
	
} // transpose

//
//	multMatrixVector()
//
//	CJS: 27/07/2015
//
//	Multiply a matrix by a vector.
//

VectorF multMatrixVector( Matrix3x3 pMatrix, VectorF pVector )
{
	
	VectorF newVector;
	
	// multiply 3x3 matrix with 3x1 vector.
	newVector.x = (pMatrix.a11 * pVector.x) + (pMatrix.a12 * pVector.y) + (pMatrix.a13 * pVector.z);
	newVector.y = (pMatrix.a21 * pVector.x) + (pMatrix.a22 * pVector.y) + (pMatrix.a23 * pVector.z);
	newVector.z = (pMatrix.a31 * pVector.x) + (pMatrix.a32 * pVector.y) + (pMatrix.a33 * pVector.z);
	
	// return something.
	return newVector;
	
} // multMatrixVector

//
//	multMatrix
//
//	CJS: 27/07/2015
//
//	Multiply two 3x3 matrices together.
//

Matrix3x3 multMatrix( Matrix3x3 pMatrix1, Matrix3x3 pMatrix2 )
{
	
	Matrix3x3 newMatrix;
	
	// row 1.
	newMatrix.a11 = (pMatrix1.a11 * pMatrix2.a11) + (pMatrix1.a12 * pMatrix2.a21) + (pMatrix1.a13 * pMatrix2.a31);
	newMatrix.a12 = (pMatrix1.a11 * pMatrix2.a12) + (pMatrix1.a12 * pMatrix2.a22) + (pMatrix1.a13 * pMatrix2.a32);
	newMatrix.a13 = (pMatrix1.a11 * pMatrix2.a13) + (pMatrix1.a12 * pMatrix2.a23) + (pMatrix1.a13 * pMatrix2.a33);
	
	// row 2.
	newMatrix.a21 = (pMatrix1.a21 * pMatrix2.a11) + (pMatrix1.a22 * pMatrix2.a21) + (pMatrix1.a23 * pMatrix2.a31);
	newMatrix.a22 = (pMatrix1.a21 * pMatrix2.a12) + (pMatrix1.a22 * pMatrix2.a22) + (pMatrix1.a23 * pMatrix2.a32);
	newMatrix.a23 = (pMatrix1.a21 * pMatrix2.a13) + (pMatrix1.a22 * pMatrix2.a23) + (pMatrix1.a23 * pMatrix2.a33);
	
	// row 3.
	newMatrix.a31 = (pMatrix1.a31 * pMatrix2.a11) + (pMatrix1.a32 * pMatrix2.a21) + (pMatrix1.a33 * pMatrix2.a31);
	newMatrix.a32 = (pMatrix1.a31 * pMatrix2.a12) + (pMatrix1.a32 * pMatrix2.a22) + (pMatrix1.a33 * pMatrix2.a32);
	newMatrix.a33 = (pMatrix1.a31 * pMatrix2.a13) + (pMatrix1.a32 * pMatrix2.a23) + (pMatrix1.a33 * pMatrix2.a33);
	
	// return something.
	return newMatrix;
	
} // multMatrix

//
//	MATRIX ROTATION FUNCTIONS
//

//
//	rotateX
//
//	CJS: 24/07/2015
//
//	Construct a 3x3 matrix to rotate a vector about the X-axis.
//

Matrix3x3 rotateX( double pAngle )
{
	
	Matrix3x3 rotationMatrix;
	
	// row 1.
	rotationMatrix.a11 = 1;
	rotationMatrix.a12 = 0;
	rotationMatrix.a13 = 0;
	
	// row 2.
	rotationMatrix.a21 = 0;
	rotationMatrix.a22 = cos( rad( pAngle ) );
	rotationMatrix.a23 = -sin( rad( pAngle ) );
	
	// row 3.
	rotationMatrix.a31 = 0;
	rotationMatrix.a32 = sin( rad( pAngle ) );
	rotationMatrix.a33 = cos( rad( pAngle ) );
	
	// return something.
	return rotationMatrix;
	
} // rotateX

//
//	rotateY
//
//	CJS: 24/07/2015
//
//	Construct a 3x3 matrix to rotate a vector about the Y-axis.
//

Matrix3x3 rotateY( double pAngle )
{
	
	Matrix3x3 rotationMatrix;
	
	// row 1.
	rotationMatrix.a11 = cos( rad( pAngle ) );
	rotationMatrix.a12 = 0;
	rotationMatrix.a13 = -sin( rad( pAngle ) );
	
	// row 2.
	rotationMatrix.a21 = 0;
	rotationMatrix.a22 = 1;
	rotationMatrix.a23 = 0;
	
	// row 3.
	rotationMatrix.a31 = sin( rad( pAngle ) );
	rotationMatrix.a32 = 0;
	rotationMatrix.a33 = cos( rad( pAngle ) );
	
	// return something.
	return rotationMatrix;
	
} // rotateY

//
//	rotateZ
//
//	CJS: 24/07/2015
//
//	Construct a 3x3 matrix to rotate a vector about the Z-axis.
//

Matrix3x3 rotateZ( double pAngle )
{
	
	Matrix3x3 rotationMatrix;
	
	// row 1.
	rotationMatrix.a11 = cos( rad( pAngle ) );
	rotationMatrix.a12 = -sin( rad( pAngle ) );
	rotationMatrix.a13 = 0;
	
	// row 2.
	rotationMatrix.a21 = sin( rad( pAngle ) );
	rotationMatrix.a22 = cos( rad( pAngle ) );
	rotationMatrix.a23 = 0;
	
	// row 3.
	rotationMatrix.a31 = 0;
	rotationMatrix.a32 = 0;
	rotationMatrix.a33 = 1;
	
	// return something.
	return rotationMatrix;
	
} // rotateZ

//
//	calculateRotationMatrix()
//
//	CJS: 03/08/2015
//
//	Calculate a rotation matrix that moves from pixel coordinates (i.e. centred on ra 0, dec 0) to the required ra and dec. The
//	inverse matrix is also calculated.
//

void calculateRotationMatrix( CoordinateSystem * pCoordinateSystem, bool pEpochConversion )
{
	
	// rotate about y-axis to bring to the correct latitude.
	pCoordinateSystem->toWorld = rotateY( pCoordinateSystem->crVAL.y );
		
	// rotate about z-axis to bring to the correct longitude.
	pCoordinateSystem->toWorld = multMatrix( rotateZ( pCoordinateSystem->crVAL.x ), pCoordinateSystem->toWorld );
		
	// do epoch conversion if required. if input and output are in the same epoch, then the epoch conversion matrix will be
	// the identity matrix.
	if (pEpochConversion == true)
		pCoordinateSystem->toWorld = multMatrix( _epochConversion, pCoordinateSystem->toWorld );
	
	// calculate the inverse as well. we'll need it to convert from world to pixel coordinates.
	pCoordinateSystem->toPixel = transpose( pCoordinateSystem->toWorld );
	
} // calculateRotationMatrix

//
//	EPOCH CONVERSION FUNCTIONS
//

//	epochConversionMatrix
//
//	CJS: 03/08/2015
//
//	Constructs a rotation matrix that converts coordinates from one epoch to another. This is done using a longitude rotation, a latitude rotation, and
//	another longitude rotation. The three rotation angles are specified as constants at the top of this program, and can be easily found using an online tool
//	such as NED (https://ned.ipac.caltech.edu/forms/calculator.html). Using NED, simply convert a position at RA 0, DEC 90 from one epoch to another and the three
//	rotation angles are given as the output coordinates (RA, DEC, PA).
//

Matrix3x3 epochConversionMatrix( double pNP_RA, double pNP_DEC, double pNP_RA_OFFSET )
{
	
	// rotate about the Z-axis by RA to bring the output north pole to RA zero.
	Matrix3x3 rotationMatrix = rotateZ( -pNP_RA );
	
	// rotate about the Y-axis by DEC to bring the output north pole to DEC 90.
	rotationMatrix = multMatrix( rotateY( 90 - pNP_DEC ), rotationMatrix );
	
	// rotate about the Z-axis by Position Angle (PA) to bring the output epoch origin to RA zero.
	rotationMatrix = multMatrix( rotateZ( pNP_RA_OFFSET ), rotationMatrix );
	
	// return something.
	return rotationMatrix;
	
} // epochConversionMatrix

//
//	doEpochConversion()
//
//	CJS: 03/08/2015
//
//	Construct a matrix that does epoch conversion between two positions. We simply compare the from and to
//	epoch, and then construct a suitable rotation matrix.
//

Matrix3x3 doEpochConversion( CoordinateSystem pFrom, CoordinateSystem pTo )
{
	
	// default to no epoch conversion.
	Matrix3x3 epochConversion = { .a11 = 1, .a12 = 0, .a13 = 0, .a21 = 0, .a22 = 1, .a23 = 0, .a31 = 0, .a32 = 0, .a33 = 1 };
	
	// J2000 to/from galactic.
	if (pFrom.epoch == EPOCH_J2000 && pTo.epoch == EPOCH_GALACTIC)
		epochConversion = epochConversionMatrix( NP_RA_GAL_IN_J2000, NP_DEC_GAL_IN_J2000, NP_RA_OFFSET_GAL_IN_J2000 );
	if (pFrom.epoch == EPOCH_GALACTIC && pTo.epoch == EPOCH_J2000)
		epochConversion = epochConversionMatrix( NP_RA_J2000_IN_GAL, NP_DEC_J2000_IN_GAL, NP_RA_OFFSET_J2000_IN_GAL );
	
	// B1950 to/from galactic.
	if (pFrom.epoch == EPOCH_B1950 && pTo.epoch == EPOCH_GALACTIC)
		epochConversion = epochConversionMatrix( NP_RA_GAL_IN_B1950, NP_DEC_GAL_IN_B1950, NP_RA_OFFSET_GAL_IN_B1950 );
	if (pFrom.epoch == EPOCH_GALACTIC && pTo.epoch == EPOCH_B1950)
		epochConversion = epochConversionMatrix( NP_RA_B1950_IN_GAL, NP_DEC_B1950_IN_GAL, NP_RA_OFFSET_B1950_IN_GAL );
	
	// B1950 to/from J2000.
	if (pFrom.epoch == EPOCH_B1950 && pTo.epoch == EPOCH_J2000)
		epochConversion = epochConversionMatrix( NP_RA_J2000_IN_B1950, NP_DEC_J2000_IN_B1950, NP_RA_OFFSET_J2000_IN_B1950 );
	if (pFrom.epoch == EPOCH_J2000 && pTo.epoch == EPOCH_B1950)
		epochConversion = epochConversionMatrix( NP_RA_B1950_IN_J2000, NP_DEC_B1950_IN_J2000, NP_RA_OFFSET_B1950_IN_J2000 );
	
	// return something.
	return epochConversion;
	
} // doEpochConversion

//
//	getEpoch()
//
//	CJS: 03/08/2015
//
//	Determine whether the epoch is J2000, B1950 or GALACTIC. Returns an enumerated type.
//

Epoch getEpoch( char * pEpoch )
{
	
	const char J2000[20] = "J2000";
	const char B1950[20] = "B1950";
	const char GALACTIC[20] = "GALACTIC";
	
	// default to J2000.
	Epoch thisEpoch = EPOCH_J2000;
	
	toUppercase( pEpoch );
	if ( strcmp( pEpoch, J2000 ) == 0 )
		thisEpoch = EPOCH_J2000;
	else if ( strcmp( pEpoch, B1950 ) == 0 )
		thisEpoch = EPOCH_B1950;
	else if ( strcmp( pEpoch, GALACTIC ) == 0 )
		thisEpoch = EPOCH_GALACTIC;
	
	// return something.
	return thisEpoch;
	
} // getEpoch

//
//	BITMAP FUNCTIONS
//

//
//	loadBitmap()
//
//	CJS: 07/07/2015
//
//	load a bitmap file, and return the image size and a boolean indicating success.
//	the image must be 8-bit greyscale.
//

bool loadBitmap( const char * pFilename, VectorI * pSize, unsigned char ** pImageData )
{
	
	bool ok = true;
	unsigned char * fileInfo, * fileHeader;
	
	// open the bitmap file.
	FILE * inputFile = fopen( pFilename, "r" );
	if (inputFile == NULL)
	{
		printf("Could not open file \"%s\".\n", pFilename);
		ok = false;
	}
	else
	{
		
		// reserve memory for the start of the file header, and read it from the file. we only
		// read the first 18 bytes, because these contain information about how large the header is. once we
		// know this we can read the rest of the header.
		fileInfo = (unsigned char *) malloc( 18 );
		size_t num_read = fread( fileInfo, sizeof( unsigned char ), 18, inputFile );
				
		// ensure we've read the correct number of bytes.
		if (num_read != 18)
		{
			printf( "Error: read only %lu values from the file header.\n", num_read );
			ok = false;
		}

		// make sure this is a bitmap file by checking that the first two bytes are ASCII codes 'B' and 'M'.
		if (ok == true)
			if ((fileInfo[BIT_CONST] != 'B') || (fileInfo[MAP_CONST] != 'M'))
			{
				printf( "Error: this is not a bitmap file.\n" );
				ok = false;
			}
			
		// get the size of the file header (i.e. a pointer to the start of the actual image).
		int fileHeaderSize = 0;
		if (ok == true)
			memcpy( &fileHeaderSize, &fileInfo[FILE_HEADER_SIZE], 4 );
			
		// get the size of the bitmap info header (the bitmap info header is followed by the colour table,
		// so we need to know the offset in order to read the colours).
		int bitmapInfoHeaderSize = 0;
		if (ok == true)
			memcpy( &bitmapInfoHeaderSize, &fileInfo[BITMAP_INFO_HEADER], 4 );
		
		// need to add 14 because the bitmap info header size does not include the first 14 bytes of the file (which
		// technically are part of the file header but not the bitmap header; we lump everything in together so that
		// all of our offsets are from the start of the file - less confusing this way).
		bitmapInfoHeaderSize = bitmapInfoHeaderSize + 14;
			
		// get the rest of the file header now we know how big it is. we already have the first 18 bytes,
		// which should be copied to the start of the new memory area.
		if (ok == true)
		{
			fileHeader = (unsigned char *) malloc( fileHeaderSize );
			memcpy( fileHeader, fileInfo, 18 );
			num_read = fread( &fileHeader[18], sizeof( unsigned char ), fileHeaderSize - 18, inputFile );
			if (num_read != (fileHeaderSize - 18))
			{
				printf( "Error: read only %lu values from the file header.\n", num_read + 18 );
				ok = false;
			}
		}
		
		// get the input image flux scale. this value may be stored in the reserved part of the bitmap file header
		// (0x06 -> 0x09), and will not be supplied if the input image has been saved using something like GIMP or
		// Photoshop. if it is zero, then we assume a scale of 1 Jy/PIXEL. this value gets re-scaled along with our
		// image, and is then written back to the output file.
		if (ok == true)
			memcpy( &_inFluxScale, &fileHeader[RESERVED], 4 );
		
		if (_inFluxScale == 0)
			_inFluxScale = 1;
			
		// ensure we have an 8-bit image.
		if (ok == true)
		{
			short bitCount;
			memcpy( &bitCount, &fileHeader[BIT_COUNT], 2 );
			if (bitCount != 8)
			{
				printf( "Error: expecting an 8-bit greyscale image. This one is %hi bit.\n", bitCount );
				ok = false;
			}
		}
			
		// ensure the image in not compressed.
		if (ok == true)
		{
			int compressionMethod;
			memcpy( &compressionMethod, &fileHeader[COMPRESSION_TYPE], 4 );
			if (compressionMethod != 0)
			{
				printf( "Error: can only handle uncompressed bitmaps." );
				ok = false;
			}
		}
			
		if (ok == true)
		{
			
			// get the width and height of the image.
			memcpy( &(pSize->x), &fileHeader[IMAGE_WIDTH], 4 );
			memcpy( &(pSize->y), &fileHeader[IMAGE_HEIGHT], 4 );
		
			// ensure width and height are greater than zero.
			if (pSize->x <= 0 || pSize->y <= 0)
			{
				printf( "Error: invalid image size (%i x %i).\n", pSize->x, pSize->y );
				ok = false;
			}
			
		}
		
		if (ok == true)
		{
			
			// ensure the number of colours used is 256.
			int coloursUsed = 0;
			memcpy( &coloursUsed, &fileHeader[COLOURS_USED], 4 );
			if (coloursUsed != 256)
			{
				printf( "ERROR: Can only handle 256 colours in pallette.\n" );
				ok = false;
			}
			
		}
		
		// get the number of significant colours used. this value can (theoretically) be less than COLOURS_USED
		// if an image is only using (e.g.) 37 shades rather than all 256. in practice, this is never implemented, and
		// this value will either be 0 (= all colours) or will match COLOURS_USED. however, only SIGNIFICANT_COLOURS are
		// written to the pallette, so we have to handle this parameter just in case.
		int significantColours = 0;
		if (ok == true)
			memcpy( &significantColours, &fileHeader[SIGNIFICANT_COLOURS], 4 );
		
		// if significant colours = 0, then they are ALL significant so set to 256.
		if (significantColours == 0)
			significantColours = 256;
			
		unsigned int colour[256];
		if (ok == true)
		{
				
			// load colour table from bmp.
			for ( unsigned int i = 0; i < significantColours; ++i )
			{
				
				memcpy( &colour[i], &fileHeader[bitmapInfoHeaderSize + (i * 4)], 4 );
				
				// convert pallette colour to greyscale, using 0.2990, 0.5870, 0.1140 RGB weighting. add 0.5
				// to round to nearest integer (since C only rounds down).
				unsigned char red = colour[i] >> 16;
				unsigned char green = (colour[i] >> 8) - (red << 8);
				unsigned char blue = colour[i] - (red << 16) - (green << 8);
				colour[i] = (unsigned int) ((((double)red * 0.2990) + ((double)green * 0.5870) +
								((double)blue * 0.1140)) + 0.5);
				
			}
				
			// reserve some memory for the image, and read it from the file.
			*pImageData = (unsigned char *) malloc( pSize->x * pSize->y );
			num_read = fread( *pImageData, sizeof( unsigned char ), pSize->x * pSize->y, inputFile );
				
			// ensure we've read the correct number of bytes.
			if (num_read != pSize->x * pSize->y)
			{
				printf( "Error: read only %lu values from the image.\n", num_read );
				ok = false;
			}
				
		}
			
		if (ok == true)
		{
				
			// update image values using the values from the colour table.
			unsigned char * imageData = *pImageData;
			for ( int i = 0; i < pSize->x * pSize->y; i++ )
				imageData[i] = (unsigned char)colour[imageData[i]];
				
		}
		
		// close file.
		fclose( inputFile );
	
	}
	
	// tidy up memory.
	if ( fileInfo != NULL )
		free( (void *) fileInfo );
	if ( fileHeader != NULL )
		free( (void *) fileHeader );
	
	// return success flag.
	return ok;
	
} // loadBitmap

//
//	saveBitmap()
//
//	CJS:	08/07/2015
//
//	write the output bitmap to file.
//

bool saveBitmap( const char * pFilename, VectorI pSize, unsigned char * image )
{
	
	const int HEADER_SIZE = 1078;
	
	// allocate and build the header.
	unsigned char * fileHeader = (unsigned char *) malloc( HEADER_SIZE );
	memset( fileHeader, 0, HEADER_SIZE );

	// file header.
	fileHeader[BIT_CONST] = 'B'; fileHeader[MAP_CONST] = 'M';					// bfType
	int size = (pSize.x * pSize.y) + HEADER_SIZE; memcpy( &fileHeader[IMAGE_SIZE], &size, 4 );	// bfSize
	int offBits = HEADER_SIZE; memcpy( &fileHeader[FILE_HEADER_SIZE], &offBits, 4 );		// bfOffBits
	
	// we write our flux scale (in Jy/PIXEL) to the reserved part of the bitmap. this
	// space would not normally be filled.
	memcpy(	&fileHeader[RESERVED], &_outFluxScale, 4 );						// bfReserved1

	// image header.
	size = 40; memcpy( &fileHeader[BITMAP_INFO_HEADER], &size, 4 );					// biSize
	memcpy( &fileHeader[IMAGE_WIDTH], &pSize.x, 4 );						// biWidth
	memcpy( &fileHeader[IMAGE_HEIGHT], &pSize.y, 4 );						// biHeight
	short planes = 1; memcpy( &fileHeader[COLOUR_PLANES], &planes, 2 );				// biPlanes
	short bitCount = 8; memcpy( &fileHeader[BIT_COUNT], &bitCount, 2 );				// biBitCount
	int coloursUsed = 256; memcpy( &fileHeader[COLOURS_USED], &coloursUsed, 4 );			// biClrUsed

	// colour table.
	for (unsigned int i = 0; i < 256; ++i)
	{
		unsigned int colour = (i << 16) + (i << 8) + i;
		memcpy( &fileHeader[54 + (i * 4)], &colour, 4 );
	}
	
	bool ok = true;

	// open file.
	FILE * outputFile = fopen( pFilename, "w" );
	if (outputFile == NULL)
	{
		printf( "Could not open file \"%s\".\n", pFilename );
		ok = false;
	}
	else
	{

		// write the file header.
		size_t num_written = fwrite( fileHeader, 1, 1078, outputFile );
		if (num_written != 1078)
		{
			printf( "Error: cannot write to file.\n" );
			ok = false;
		}
		
		// write the data.
		if (ok == true)
		{
			
			size_t num_written = fwrite( image, 1, pSize.x * pSize.y, outputFile );
			if (num_written != (pSize.x * pSize.y))
			{
				printf( "Error: cannot write to file.\n" );
				ok = false;
			}
			
		}

		// close file.
		fclose( outputFile );
		
	}

	// cleanup memory.
	free( (void *) fileHeader );
	
	// return success flag.
	return ok;
	
} // saveBitmap

//
//	COORDINATE SYSTEM TRANSFORMATION FUNCTIONS
// 

//
//	pixelToWorld()
//
//	CJS:	08/07/15
//
//	Convert pixel coordinates into world coordinates using the supplied coordinate system. CASA uses WCSLIB to do this,
//	which does a complete spherical transformation of the coordinates. However, here we compromise and just use the
//	FITS matrix transformation (CD) to do a linear conversion from pixel to intermediate coordinates, and then multiply
//	the first coordinate by cos(dec) in order to convert from an angular size to degrees of RA (i.e. implement sinusoidal
//	projection).
//
//	The rotation matrix attached to the coordinate system will convert the coordinates from the origin (chosen to be
//	RA 0, DEC 0) to the relevant RA and DEC position of the reference pixel. Epoch conversion will be done if required.
//
//	Note that the routine returns the new position in cartesian coordinates (using directional cosines). This is because
//	the world to pixel routine needs cartesian coordinates.
//
//	The returned 'wrap around' flag warns if this pixel is outside the range -180 to 180 in RA, or -90 to 90 in DEC.
//

VectorF	pixelToWorld( VectorF pPixelPosition, CoordinateSystem pCoordinateSystem, bool * pWrapAround )
{
	
	// subtract reference pixel from position.
	VectorF pixelOffset;
	pixelOffset.x = pPixelPosition.x - pCoordinateSystem.crPIX.x;
	pixelOffset.y = pPixelPosition.y - pCoordinateSystem.crPIX.y;
	
	// apply coordinate system CD transformation matrix.
	VectorF intermediatePosition;
	intermediatePosition.x = (pCoordinateSystem.cd.a11 * pixelOffset.x) + (pCoordinateSystem.cd.a12 * pixelOffset.y);
	intermediatePosition.y = (pCoordinateSystem.cd.a21 * pixelOffset.x) + (pCoordinateSystem.cd.a22 * pixelOffset.y);
	
	// skew the image using the declination. this step reflects the fact that lines of RA get closer together
	// as the image gets nearer to +/- 90 deg declination. this transformation effectively converts from angular
	// distance in the ra direction to actual ra coordinates.
	VectorF worldOffset;
	worldOffset.x = intermediatePosition.x / cos( rad( intermediatePosition.y ));
	worldOffset.y = intermediatePosition.y;
	
	// check for wrap around, and set the flag.
	if (worldOffset.x < -180 || worldOffset.x > 180 || worldOffset.y < -90 || worldOffset.y > 90)
		*pWrapAround = true;
	
	// get x, y and z cartesian coordinates.
	VectorF cartesianOffset;
	cartesianOffset.x = cos( rad( worldOffset.x ) ) * cos( rad( worldOffset.y ) );
	cartesianOffset.y = sin( rad( worldOffset.x ) ) * cos( rad( worldOffset.y ) );
	cartesianOffset.z = sin( rad( worldOffset.y ) );
	
	// the world offset coordinates are relative to the reference pixel, which is currently at ra 0, dec 0. we need to
	// rotate the offset coordinates by the reference pixel's true ra and dec so that they are relative to ra 0, dec 0.
	// unfortunately, the dec rotation has to be done in cartesian coordinates, which are rather messy to convert back
	// to spherical.
	cartesianOffset = multMatrixVector( pCoordinateSystem.toWorld, cartesianOffset );
	
	// return the world position.
	return cartesianOffset;
	
} // pixelToWorld

//
//	worldToPixel()
//
//	CJS:	08/07/15
//
//	Convert world coordinates into pixel coordinates using the supplied coordinate system. Now we must use
//	the inverse transformation matrix, which was calculated earlier from CD.
//

VectorF worldToPixel( VectorF pWorldPosition, CoordinateSystem pCoordinateSystem )
{
	
	// rotate the vector to bring it from its world position near RA and DEC to the origin, which we choose
	// to be RA 0, DEC 0.
	VectorF cartesianOffset = multMatrixVector( pCoordinateSystem.toPixel, pWorldPosition );
	
	// we now need to convert back into polar coordinates.
	VectorF intermediatePosition = { .x = arctan( cartesianOffset.y, cartesianOffset.x ), .y = deg( asin( cartesianOffset.z ) ) };
	
	// ensure right ascention is within the required range (-180 to 180 degrees).
	angleRange( &intermediatePosition.x, 0 );
	
	// skew the image using the declination. this step reflects the fact that lines of RA get closer together
	// as the image gets nearer to +/- 90 deg declination.
	intermediatePosition.x = intermediatePosition.x * cos( rad( intermediatePosition.y ) );
	
	// apply coordinate system inverse-CD transformation matrix.
	VectorF pixelOffset;
	pixelOffset.x = (pCoordinateSystem.inv_cd.a11 * intermediatePosition.x) +
				(pCoordinateSystem.inv_cd.a12 * intermediatePosition.y);
	pixelOffset.y = (pCoordinateSystem.inv_cd.a21 * intermediatePosition.x) +
				(pCoordinateSystem.inv_cd.a22 * intermediatePosition.y);
	
	// add reference pixel coordinates.
	VectorF pixelPosition;
	pixelPosition.x = pixelOffset.x + pCoordinateSystem.crPIX.x;
	pixelPosition.y = pixelOffset.y + pCoordinateSystem.crPIX.y;
	
	// return the world position.
	return pixelPosition;
	
} // worldToPixel

//
//	REPROJECTION AND REGRIDDING FUNCTIONS
//

//
//	interpolateValue()
//
//	CJS:	08/07/15
//
//	use 'pPosition' to do bilinear interpolation between 4 data points.
//

double interpolateValue( VectorF pPosition, double pBLValue, double pBRValue, double pTLValue, double pTRValue )
{
	
	// subtract the integer part of the position. we don't need this here.
	VectorI integerPart = { .x = (int) floor( pPosition.x ), .y = (int) floor( pPosition.y ) };
	VectorF fraction = { .x = pPosition.x - (double)integerPart.x, .y = pPosition.y - (double)integerPart.y };
		
	// interpolate top and bottom in the x-direction.
	double valueTop = ((pTRValue - pTLValue) * fraction.x) + pTLValue;
	double valueBottom = ((pBRValue - pBLValue) * fraction.x) + pBLValue;
		
	// interpolate in y-direction.
	return ((valueTop - valueBottom) * fraction.y) + valueBottom;
	
} // interpolateValue

//
//	getParameters()
//
//	CJS: 07/07/2015
//
//	Load the parameters from the parameter file 'reprojection-params'.
//

void getParameters()
{

	char params[80], line[1024], par[80];
 
	// Open the parameter file and get all lines.
	FILE *fr = fopen( "reprojection-params", "rt" );
	while ( fgets(line, 80, fr) != NULL )
	{

		sscanf( line, "%s %s", par, params );
		if ( strcmp( par, IN_CRVAL1 ) == 0)
			_inCoordinateSystem.crVAL.x = atof( params );
		else if ( strcmp( par, IN_CRVAL2 ) == 0 )
			_inCoordinateSystem.crVAL.y = atof( params );
		else if ( strcmp( par, IN_CRPIX1 ) == 0 )
			_inCoordinateSystem.crPIX.x = atoi( params );
		else if ( strcmp( par, IN_CRPIX2 ) == 0 )
			_inCoordinateSystem.crPIX.y = atoi( params );
		else if ( strcmp( par, IN_CD1_1 ) == 0 )
			_inCoordinateSystem.cd.a11 = atof( params );
		else if ( strcmp( par, IN_CD1_2 ) == 0 )
			_inCoordinateSystem.cd.a12 = atof( params );
		else if ( strcmp( par, IN_CD2_1 ) == 0 )
			_inCoordinateSystem.cd.a21 = atof( params );
		else if ( strcmp( par, IN_CD2_2 ) == 0 )
			_inCoordinateSystem.cd.a22 = atof( params );
		else if ( strcmp( par, IN_EPOCH ) == 0 )
			_inCoordinateSystem.epoch = getEpoch( params );
		else if ( strcmp ( par, OUT_SIZE_1 ) == 0 )
			_outSize.x = atoi( params );
		else if ( strcmp ( par, OUT_SIZE_2 ) == 0 )
			_outSize.y = atoi( params );
		else if ( strcmp( par, OUT_CRVAL1 ) == 0)
			_outCoordinateSystem.crVAL.x = atof( params );
		else if ( strcmp( par, OUT_CRVAL2 ) == 0 )
			_outCoordinateSystem.crVAL.y = atof( params );
		else if ( strcmp( par, OUT_CRPIX1 ) == 0 )
			_outCoordinateSystem.crPIX.x = atoi( params );
		else if ( strcmp( par, OUT_CRPIX2 ) == 0 )
			_outCoordinateSystem.crPIX.y = atoi( params );
		else if ( strcmp( par, OUT_CD1_1 ) == 0 )
			_outCoordinateSystem.cd.a11 = atof( params );
		else if ( strcmp( par, OUT_CD1_2 ) == 0 )
			_outCoordinateSystem.cd.a12 = atof( params );
		else if ( strcmp( par, OUT_CD2_1 ) == 0 )
			_outCoordinateSystem.cd.a21 = atof( params );
		else if ( strcmp( par, OUT_CD2_2 ) == 0 )
			_outCoordinateSystem.cd.a22 = atof( params );
		else if ( strcmp( par, OUT_EPOCH ) == 0 )
			_outCoordinateSystem.epoch = getEpoch( params );
            
	}
	fclose(fr);

} // getParameters

//
//	reprojection()
//
//	CJS: 07/07/2015
//
//	Performs regridding and reprojection between the input and output images.
//	We need to handle the case where the output image pixels are much larger than the input image pixels (we need
//	to sum over many pixels), and also when the output image pixels are much smaller than the input image pixels
//	(we need to interpolate between input image pixels).
//
//	This routine works by comparing the size of the input and output image pixels, and choosing a number of
//	interpolation points for each output pixel. For example, overlaying the input and output images in world
//	coordinates may give:
//
//		+--------+--------+--------+--------+
//		|        |        |        |        |	+---+
//		|        |#   #   #   #   #|  #     |	|   |	= input image pixels
//		|        |        |                 |	+---+
//		+--------+#-------+--------+--#-----+
//		|        |     X====X====X |        |	+===+
//		|        |#    I  |      I |  #     |	I   I	= output image pixel
//		|        |     X  | X    X |        |	+===+
//		+--------+#----I--+------I-+--#-----+
//		|        |     X====X====X |        |	# # #	  region centred on the output image pixel, that extends on
//		|        |#       |        |  #     |	#   #	= all four sides to the surrounding output image pixels. this
//		|        |        |        |        |	# # #	  is the region we sum over.
//		+--------+#---#---#---#---#+--#-----+
//		|        |        |        |        |	  X	= interpolation point. the centre point has weight 1, the ones
//		|        |        |        |        |					along side it have weight 0.5, and the
//		|        |        |        |        |					ones on the diagonals have weight 0.25.
//		+--------+--------+--------+--------+
//
//	The program uses bilinear interpolation to calculate the value of the input grid at each interpolation point. These
//	values are then summed using a weighting that depends upon the position of the interpolation point relative to the output
//	pixel (the centre of the output pixel has weight 1, and this drops to 0 as we near the adjacent output pixels). If the
//	output pixel is small compared to the input pixels then we use a small number of interpolation points (one would do the
//	job, but we use a minimum of 3x3). If the output pixel is large compared to the input pixels then we use many
//	interpolation points (enough to ensure that at least one interpolation point is found within each fully-enclosed input
//	pixel).
//

void reprojection()
{
	
	const int POS_BL = 0;
	const int POS_BR = 1;
	const int POS_TL = 2;
	const int POS_TR = 3;
	
	// reserve memory for the output image.
	_outputImage = (unsigned char *) malloc( _outSize.x * _outSize.y );
	
	// report progress because this could take a long time.
	printf( "Calculating....." );
	fflush( stdout );
	
	// loop through all the output image pixels.
	int fraction = -1;
	for ( int i = 0; i < _outSize.x; i++ )
	{
		
		// display progress.
		if ((double)i / (double)_outSize.x >= (double)(fraction + 1) / 20)
		{
			fraction = fraction + 1;
			printf( "%i%%.", fraction * 5 );
			fflush( stdout );
		}
		
		for ( int j = 0; j < _outSize.y; j++ )
		{
			
			int pixelValue = DEFAULT_PIXEL_VALUE;
			
			// we need to find the coordinates of the 4 output pixels that diagonally surround our
			// pixel, so that we can choose some interpolation points within this region of the input image.
			VectorF outPixelCoordinate[4];
			VectorF worldCoordinate[4];
			VectorF inPixelCoordinate[4];
			
			outPixelCoordinate[POS_BL].x = i - 1; outPixelCoordinate[POS_BL].y = j - 1;
			outPixelCoordinate[POS_BR].x = i + 1; outPixelCoordinate[POS_BR].y = j - 1;
			outPixelCoordinate[POS_TL].x = i - 1; outPixelCoordinate[POS_TL].y = j + 1;
			outPixelCoordinate[POS_TR].x = i + 1; outPixelCoordinate[POS_TR].y = j + 1;
			
			// convert each of these four pixels to the input coordinate system.
			bool wrapAround = false;
			for ( int k = 0; k < 4; k++ )
			{
					
				// convert these pixel coordinates to world coordinates (cartesian).
				worldCoordinate[k] = pixelToWorld( outPixelCoordinate[k], _outCoordinateSystem, &wrapAround );
			
				// convert the world coordinates back into input pixel coordinates.
				inPixelCoordinate[k] = worldToPixel( worldCoordinate[k], _inCoordinateSystem );
					
			}
			
			// if we have wrapped around, then leaves this pixel black. Otherwise, carry on.
			if (wrapAround == false)
			{
				
				// the input pixel coordinates will map out same shape in the input image, which will not necessarily
				// be square. we need to find the limits in both x and y, so that we can choose an appropriate
				// number of interpolation points within the region.
				VectorF min = inPixelCoordinate[POS_BL], max = inPixelCoordinate[POS_BL];
				for ( int k = 1; k < 4; k++ )
				{
				
					min.x = minF( min.x, inPixelCoordinate[k].x );
					max.x = maxF( max.x, inPixelCoordinate[k].x );
					min.y = minF( min.y, inPixelCoordinate[k].y );
					max.y = maxF( max.y, inPixelCoordinate[k].y );
				}
			
				// find the size of the input image region.
				int regionSize = maxI(	(int) floor( max.x ) - (int) floor( min.x ),
							(int) floor( max.y ) - (int) floor( min.y ) );
			
				// the input image pixels could be much larger than our region, or they could be much smaller.
				// we use the region size to define a number of interpolation points, which form a NxN grid
				// around our region of the input image. Each input pixel should have at least one interpolation
				// point.
				int interpolationPoints = (((int)(regionSize / 2)) + 1) * 4;
				if (interpolationPoints < 2)
					interpolationPoints = 2;
				if (interpolationPoints > MAX_INTERPOLATION_POINTS)
					interpolationPoints = MAX_INTERPOLATION_POINTS;
			
				// keep track of the total value and total weight, so we can normalise the sum over
				// the interpolation points.
				double totalValue = 0;
				double totalWeight = 0;
			
				// loop through all the interpolation points. we don't bother with the first or last
				// interpolation points, since they have weight 0.
				for ( int k = 1; k < interpolationPoints; k++ )
					for ( int l = 1; l < interpolationPoints; l++ )
					{
					
						// calculate the position of this interpolation point as a fraction
						// of the output pixel size (the centre of the pixel will be at <0.5, 0.5>).
						VectorF fraction = {	.x = (double)k / (double)interpolationPoints,
									.y = (double)l / (double)interpolationPoints };
					
						// calculate the weight of this interpolation point. this is based upon its
						// position - the centre of the region has weight 1, and the edges weight 0.
						VectorF weight = {	.x = (fraction.x <= 0.5) ? fraction.x * 2 : 2 - (fraction.x * 2),
									.y = (fraction.y <= 0.5) ? fraction.y * 2 : 2 - (fraction.y * 2) };
					
						double interpolationWeight = (weight.x * weight.y);
					
						// get the position of the interpolation point in the output image.
						VectorF outPixelInterpolationPoint = {	.x = (double)(i - 1) + (fraction.x * 2),
											.y = (double)(j - 1) + (fraction.y * 2) };
					
						// convert these pixel coordinates to world coordinates (cartesian). we ignore the wrap around
						// since we've already checked that this is OK whilst calculating the interpolation points.
						VectorF worldInterpolationPoint = pixelToWorld(	outPixelInterpolationPoint,
												_outCoordinateSystem, &wrapAround );
			
						// convert the world coordinates back into input pixel coordinates.
						VectorF inPixelInterpolationPoint = worldToPixel(	worldInterpolationPoint,
													_inCoordinateSystem );
					
						// calculate the four pixel coordinates surrounding this interpolation point.
						VectorI pixel[4];
						pixel[POS_BL].x = (int) floor( inPixelInterpolationPoint.x );
						pixel[POS_BL].y = (int) floor( inPixelInterpolationPoint.y );
						pixel[POS_BR].x = pixel[POS_BL].x + 1; pixel[POS_BR].y = pixel[POS_BL].y;
						pixel[POS_TL].x = pixel[POS_BL].x; pixel[POS_TL].y = pixel[POS_BL].y + 1;
						pixel[POS_TR].x = pixel[POS_BL].x + 1; pixel[POS_TR].y = pixel[POS_BL].y + 1;
					
						// ensure all pixels are within the extent of the input image.
						bool withinRange = true;
						for ( int i = 0; i < 4; i++ )
						{
							withinRange = withinRange && (pixel[i].x >= 0) && (pixel[i].x < _inSize.x);
							withinRange = withinRange && (pixel[i].y >= 0) && (pixel[i].y < _inSize.y);
						}
					
						// calculate memory location of this pixel within the input image.
						int location[4];
						for ( int i = 0; i < 4; i++ )
							location[i] = (pixel[i].y * _inSize.x) + pixel[i].x;
					
						// are these pixels all within the input image size?
						if (withinRange == true)
						{
										
							// get an bilinearly interpolated value from the input pixel image.
							double value = interpolateValue(	inPixelInterpolationPoint,
												(double)_inputImage[ location[POS_BL] ],
												(double)_inputImage[ location[POS_BR] ],
												(double)_inputImage[ location[POS_TL] ],
												(double)_inputImage[ location[POS_TR] ] );
												
							// update the summed value and the summed weight.
							totalValue = totalValue + (value * interpolationWeight);
							totalWeight = totalWeight + interpolationWeight;
											
						}
											
					}
				
				// calculate output pixel value.
				if (totalWeight != 0)
					pixelValue = (int)(totalValue / totalWeight);
				
				// maximum value is (currently) 255.
				pixelValue = minI( 255, pixelValue );
			
			}
				
			// update output pixel value.
			_outputImage[ (j * _outSize.x) + i ] = (unsigned char)pixelValue;
			
		}
		
	}
	printf("100%%\n");
		
	// Finally, we need to set the output flux scale to reflect the fact that input pixels may be much larger or smaller
	// than output pixels when viewed in world coordinates - so far we've just averaged/interpolated over input pixels
	// to get the output pixel values, and now we need to scale our results to reflect the change in pixel size. The
	// simplest way to do this is to compare the two coordinate systems, although in doing so we are assuming that
	// the sky is flat over the extent of the image. I notice that CASA uses the same method for correcting flux scale.
		
	// calculate the width of 1 pixel in world coordinates (i.e. degrees).
	double inPixelWidth = sqrt( pow( _inCoordinateSystem.cd.a11, 2 ) + pow( _inCoordinateSystem.cd.a21, 2 ) );
	double outPixelWidth = sqrt( pow( _outCoordinateSystem.cd.a11, 2 ) + pow( _outCoordinateSystem.cd.a21, 2 ) );
		
	// calculate the height of 1 pixel in world coordinates (i.e. degrees).
	double inPixelHeight = sqrt( pow( _inCoordinateSystem.cd.a12, 2 ) + pow( _inCoordinateSystem.cd.a22, 2 ) );
	double outPixelHeight = sqrt( pow( _outCoordinateSystem.cd.a12, 2 ) + pow( _outCoordinateSystem.cd.a22, 2 ) );
		
	// calculate the area of 1 pixel in world coordinates (i.e. degrees^2).
	double inPixelArea = inPixelWidth * inPixelHeight;
	double outPixelArea = outPixelWidth * outPixelHeight;
	
	// Calculate flux density scale for output image. As we are storing in a 8-bit bitmap, the pallette ranges from 0 to 255.
	// The flux scale gives the constant that each pixel value should be multiplied by to recover the flux density of that
	// pixel. If we were using an image format that allowed the flux density to be stored (rather than bitmap, which only
	// stores 0-255) then we would instead multiply each pixel in the output image by outPixelArea / inPixelArea. This
	// calculation could be done separately for each pixel, which would mean we were no longer using the flat-sky
	// approximation.
	_outFluxScale = _inFluxScale * outPixelArea / inPixelArea;
	
	printf( "output image flux scale %f Jy/PIXEL\n", _outFluxScale );
	
} // reprojection

//
//	main()
//
//	CJS: 07/07/2015
//
//	Load the parameters from the parameter file 'reprojection-params'.
//	Load the input image.
//	Kick off the regridding and reprojection.
//	Save the new bitmap image.
//

int main( int pArgc, char ** pArgv )
{
	
	// read program arguments. we expect to see the program call (0), the input filename (1) and the output filename (2).
	if (pArgc != 3)
	{
		printf("Wrong number of arguments. Require the input and outfile BMP filenames.\n");
		return 1;
	}
	const char * inputFilename = pArgv[1];
	const char * outputFilename = pArgv[2];

	// get the run-time parameters, which specify the input and output coordinate systems in FITS convention.
	// note that no validation is performed on these parameters, so care must be taken to get them right.
	getParameters();
	
	// prepare the epoch conversion matrix. we convert from the input epoch to the output epoch because we have decided that all the world
	// coordinates in this program will use the output epoch.
	_epochConversion = doEpochConversion( _inCoordinateSystem, _outCoordinateSystem );
	
	// calculate the rotation matrices to convert from pixel coordinates (centred at ra 0, dec 0) to world coordinates at the required ra, dec.
	// we don't want to do this every time because that would be slow and unnecessary. we only do epoch conversion for the output coordinate system
	// conversion, because we have decided that everything in this program will use the input coordinate system epoch.
	calculateRotationMatrix( &_inCoordinateSystem, true );
	calculateRotationMatrix( &_outCoordinateSystem, false );
	
	// calculate the inverse of the input and output CD transformation matrices. this inverse will be needed
	// when we transform from world to pixel coordinates.
	_inCoordinateSystem.inv_cd = calculateInverseMatrix( _inCoordinateSystem.cd );
	_outCoordinateSystem.inv_cd = calculateInverseMatrix( _outCoordinateSystem.cd );
	
	// load the input file.
	bool ok = loadBitmap( inputFilename, &_inSize, &_inputImage );
	if (ok == true)
	{
		
		// do the reprojection and regridding.
		reprojection();
		
		// save the output file.
		ok = saveBitmap( outputFilename, _outSize, _outputImage );
	
		if (ok == true)
			printf("regridding and reprojection done\n");
		else
			printf("bitmap was loaded, but result could not be saved\n");
		
	}
	else
		printf("bmp not successfully loaded\n");
	
	// tidy up memory.
	if (_inputImage != NULL)
		free( (void *) _inputImage );
	if (_outputImage != NULL)
		free( (void *) _outputImage );

	return 0;
  
} // main
