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

//
//	Structures.
//

// vector with floats.
struct VectorF
{
	double x;
	double y;
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
struct Matrix
{
	double c1_1;
	double c1_2;
	double c2_1;
	double c2_2;
};
typedef struct Matrix Matrix;

// stores a coordinate system - reference pixel coordinates and world coordinates, and transformation matrix CD.
struct CoordinateSystem
{
	VectorF crVAL;
	VectorI crPIX;
	Matrix cd;
	Matrix inv_cd;
};
typedef struct CoordinateSystem CoordinateSystem;

//
//	Constants.
//

const double PI = 3.14159265359;

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
//	Global variables.
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
		result = (pValueTop >= 0) ? 90 : 270;
	
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
//	calculateInverseMatrix()
//
//	CJS:	08/07/2015
//
//	calculate the inverse of a 2x2 matrix.
//

Matrix calculateInverseMatrix( Matrix pMatrix )
{
	
	// calculate the determinant.
	double determinant = (pMatrix.c1_1 * pMatrix.c2_2) - (pMatrix.c1_2 * pMatrix.c2_1);
		
	// create an empty matrix.
	Matrix inverse = { .c1_1 = 0, .c1_2 = 0, .c2_1 = 0, .c2_2 = 0 };
	
	// the matrix is supposed to be invertible, but we should check anyway.
	if (determinant != 0)
	{
		inverse.c1_1 = pMatrix.c2_2 / determinant;
		inverse.c1_2 = -pMatrix.c1_2 / determinant;
		inverse.c2_1 = -pMatrix.c2_1 / determinant;
		inverse.c2_2 = pMatrix.c1_1 / determinant;
	}
	
	// return the inverse matrix.
	return inverse;
	
} // calculateInverseMatrix

//
//	pixelToWorld()
//
//	CJS:	08/07/15
//
//	Convert pixel coordinates into world coordinates using the supplied coordinate system. CASA uses WCSLIB to do this,
//	which does a complete spherical transformation of the coordinates. However, here we compromise and just use the
//	FITS matrix transformation (CD) to do a linear conversion from pixel to intermediate coordinates, and then multiply
//	the first coordinate by cos(dec) in order to convert from an angular size to degrees of RA.
//

VectorF	pixelToWorld( VectorF pPixelPosition, CoordinateSystem pCoordinateSystem )
{
	
	// subtract reference pixel from position.
	VectorF pixelOffset;
	pixelOffset.x = pPixelPosition.x - pCoordinateSystem.crPIX.x;
	pixelOffset.y = pPixelPosition.y - pCoordinateSystem.crPIX.y;
	
	// apply coordinate system CD transformation matrix.
	VectorF intermediatePosition;
	intermediatePosition.x = (pCoordinateSystem.cd.c1_1 * pixelOffset.x) + (pCoordinateSystem.cd.c1_2 * pixelOffset.y);
	intermediatePosition.y = (pCoordinateSystem.cd.c2_1 * pixelOffset.x) + (pCoordinateSystem.cd.c2_2 * pixelOffset.y);
	
	// skew the image using the declination. this step reflects the fact that lines of RA get closer together
	// as the image gets nearer to +/- 90 deg declination. this transformation effectively converts from angular
	// distance in the ra direction to actual ra coordinates.
	VectorF worldOffset;
	worldOffset.x = rad( intermediatePosition.x / cos( rad( intermediatePosition.y )) ) ;
	worldOffset.y = rad( intermediatePosition.y );
	
	// store some values to avoid calculating trig terms more than once.
	VectorF cosWorldOffset = { .x = cos( worldOffset.x ), .y = cos( worldOffset.y ) };
	VectorF sinWorldOffset = { .x = sin( worldOffset.x ), .y = sin( worldOffset.y ) };
	double sinDec = sin( rad( pCoordinateSystem.crVAL.y ) );
	double cosDec = cos( rad( pCoordinateSystem.crVAL.y ) );
	
	// the world offset coordinates are relative to the reference pixel, which is currently at ra 0, dec 0. we need to
	// rotate the offset coordinates by the reference pixel's true ra and dec so that they are relative to ra 0, dec 0.
	// unfortunately, the dec rotation has to be done in cartesian coordinates, which are rather messy to convert back
	// to spherical.
	VectorF worldPosition;
	worldPosition.x = arctan( sinWorldOffset.x * cosWorldOffset.y,
				  ((cosWorldOffset.x * cosWorldOffset.y * cosDec ) - (sinWorldOffset.y * sinDec)) )
				+ pCoordinateSystem.crVAL.x;
	worldPosition.y = deg( asin( (cosWorldOffset.x * cosWorldOffset.y * sinDec) + (sinWorldOffset.y * cosDec) ) );
	
	// return the world position.
	return worldPosition;
	
} // pixelToWorld

//
//	worldToPixel()
//
//	CJS:	08/07/15
//
//	Convert world coordinates into pixel coordinates using the supplied coordinate system. Now we must use
//	the inverse transformation matrix, which was calculated earlier from CD.
//

VectorF worldToPixel( VectorF pWorldPosition, CoordinateSystem pCoordinateSystem, double pCentreRARange )
{
	
	// rotate the world coordinates by the reference pixel's true ra and dec in order that the reference pixel is
	// moved to ra 0, dec 0. Unfortunately, the dec rotation has to be done in cartesian coordinates. Start by
	// rotating to ra 0.
	VectorF worldOffset1 = { .x = rad( pWorldPosition.x - pCoordinateSystem.crVAL.x ), .y = rad( pWorldPosition.y) };
	
	// store some values to avoid calculating trig terms more than once.
	VectorF cosWorldOffset1 = { .x = cos( worldOffset1.x ), .y = cos( worldOffset1.y ) };
	VectorF sinWorldOffset1 = { .x = sin( worldOffset1.x ), .y = sin( worldOffset1.y ) };
	double sinDec = sin( rad( -pCoordinateSystem.crVAL.y ) );
	double cosDec = cos( rad( -pCoordinateSystem.crVAL.y ) );
	
	// now rotate to dec 0.
	VectorF worldOffset2;
	worldOffset2.x = arctan( sinWorldOffset1.x * cosWorldOffset1.y,
				  ((cosWorldOffset1.x * cosWorldOffset1.y * cosDec ) - (sinWorldOffset1.y * sinDec)) );
	worldOffset2.y = deg( asin( (cosWorldOffset1.x * cosWorldOffset1.y * sinDec) + (sinWorldOffset1.y * cosDec) ) );
	
	// ensure right ascention is within the required range.
	angleRange( &worldOffset2.x, pCentreRARange );
	
	// skew the image using the declination. this step reflects the fact that lines of RA get closer together
	// as the image gets nearer to +/- 90 deg declination.
	VectorF intermediatePosition;
	intermediatePosition.x = worldOffset2.x * cos( rad( worldOffset2.y ) );
	intermediatePosition.y = worldOffset2.y;
	
	// apply coordinate system inverse-CD transformation matrix.
	VectorF pixelOffset;
	pixelOffset.x = (pCoordinateSystem.inv_cd.c1_1 * intermediatePosition.x) +
				(pCoordinateSystem.inv_cd.c1_2 * intermediatePosition.y);
	pixelOffset.y = (pCoordinateSystem.inv_cd.c2_1 * intermediatePosition.x) +
				(pCoordinateSystem.inv_cd.c2_2 * intermediatePosition.y);
	
	// add reference pixel coordinates.
	VectorF pixelPosition;
	pixelPosition.x = pixelOffset.x + pCoordinateSystem.crPIX.x;
	pixelPosition.y = pixelOffset.y + pCoordinateSystem.crPIX.y;
	
	// return the world position.
	return pixelPosition;
	
} // worldToPixel

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
			_inCoordinateSystem.cd.c1_1 = atof( params );
		else if ( strcmp( par, IN_CD1_2 ) == 0 )
			_inCoordinateSystem.cd.c1_2 = atof( params );
		else if ( strcmp( par, IN_CD2_1 ) == 0 )
			_inCoordinateSystem.cd.c2_1 = atof( params );
		else if ( strcmp( par, IN_CD2_2 ) == 0 )
			_inCoordinateSystem.cd.c2_2 = atof( params );
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
			_outCoordinateSystem.cd.c1_1 = atof( params );
		else if ( strcmp( par, OUT_CD1_2 ) == 0 )
			_outCoordinateSystem.cd.c1_2 = atof( params );
		else if ( strcmp( par, OUT_CD2_1 ) == 0 )
			_outCoordinateSystem.cd.c2_1 = atof( params );
		else if ( strcmp( par, OUT_CD2_2 ) == 0 )
			_outCoordinateSystem.cd.c2_2 = atof( params );
            
	}
	fclose(fr);

} // getParameters

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
	
	// loop through all the output image pixels.
	for ( int i = 0; i < _outSize.x; i++ )
		for ( int j = 0; j < _outSize.y; j++ )
		{
			
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
			for ( int k = 0; k < 4; k++ )
			{
					
				// convert these pixel coordinates to world coordinates.
				worldCoordinate[k] = pixelToWorld( outPixelCoordinate[k], _outCoordinateSystem );
				
				// ensure that the ra of the first pixel is in range -180 -> +180, and ensure that the ra of all
				// other pixels is in a range centred on the first pixel.
				angleRange( &worldCoordinate[k].x, (k == 0) ? 0 : worldCoordinate[POS_BL].x );
			
				// convert the world coordinates back into input pixel coordinates.
				inPixelCoordinate[k] = worldToPixel( worldCoordinate[k], _inCoordinateSystem,
									worldCoordinate[POS_BL].x );
					
			}
				
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
					
					// convert these pixel coordinates to world coordinates.
					VectorF worldInterpolationPoint = pixelToWorld(	outPixelInterpolationPoint,
											_outCoordinateSystem );
			
					// convert the world coordinates back into input pixel coordinates.
					VectorF inPixelInterpolationPoint = worldToPixel(	worldInterpolationPoint,
												_inCoordinateSystem,
												_inCoordinateSystem.crVAL.x );
					
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
			int pixelValue = 0;
			if (totalWeight != 0)
				pixelValue = (int)(totalValue / totalWeight);
				
			// maximum value is (currently) 255.
			pixelValue = minI( 255, pixelValue );
				
			// update output pixel value.
			_outputImage[ (j * _outSize.x) + i ] = (unsigned char)pixelValue;
			
		}
		
	// Finally, we need to set the output flux scale to reflect the fact that input pixels may be much larger or smaller
	// than output pixels when viewed in world coordinates - so far we've just averaged/interpolated over input pixels
	// to get the output pixel values, and now we need to scale our results to reflect the change in pixel size. The
	// simplest way to do this is to compare the two coordinate systems, although in doing so we are assuming that
	// the sky is flat over the extent of the image. I notice that CASA uses the same method for correcting flux scale.
		
	// calculate the width of 1 pixel in world coordinates (i.e. degrees).
	double inPixelWidth = sqrt( pow( _inCoordinateSystem.cd.c1_1, 2 ) + pow( _inCoordinateSystem.cd.c2_1, 2 ) );
	double outPixelWidth = sqrt( pow( _outCoordinateSystem.cd.c1_1, 2 ) + pow( _outCoordinateSystem.cd.c2_1, 2 ) );
		
	// calculate the height of 1 pixel in world coordinates (i.e. degrees).
	double inPixelHeight = sqrt( pow( _inCoordinateSystem.cd.c1_2, 2 ) + pow( _inCoordinateSystem.cd.c2_2, 2 ) );
	double outPixelHeight = sqrt( pow( _outCoordinateSystem.cd.c1_2, 2 ) + pow( _outCoordinateSystem.cd.c2_2, 2 ) );
		
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
