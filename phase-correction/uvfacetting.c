//
//	uvfacetting.c
//
//	Chris Skipper
//	24/07/2015
//
//	Changing the phase position on the sky from the initial phase position (or pointing position of the telescope) can be done
//	in two ways. The first (which we call UV facetting), applies a phase correction to each visibility in order that the phase centre
//	is shifted around the celestial sphere to a new position. This method is initiated by setting the 'uv_projection' parameter
//	in the 'uvfacetting-params' file to 'N'.
//
//	The second method (which we call image plane facetting) also uses a phase correction, but in addition will correct the
//	uvw coordinates in order that the resulting image is projected onto a new tangential plane centred on the new phase position. This
//	method is initiated by setting the 'uv_projection' parameter to 'Y'. We use the equations from Sault et al 1996 A&As 120 375-384 to
//	do this reprojection.
//
//                Image plane facetting                       UV facetting
//
//              ---------/---------------                   ----------------
//                     / oo IN oo                               oo IN oo
//                   / oo        ooo                    OUT  ooo        ooo
//                 / OUT            oo            ----------------         oo
//               / o                  o                   o                  o
//             /  o                    o                 o                    o
//                o                    o                 o                    o
//
//	The input and output UVW coordinate systems are aligned such that w points directly at the relevant phase position, and u points
//	directly east when the phase centre is at the local meridian, and the v-axis completes the cross product.
//
//	This program rotates the coordinate systems, and the supplied uvw vector, such that either the input or output coordinate systems
//	or the coordinate system of the celestial sphere can be aligned with the chosen xyz cartesian axes.
//
//	The UVW coordinate systems can be aligned with xyz such that u is along the x-axis, v is along the y-axis, and w is along the z-axis.
//	When the coordinate system of the celestial sphere is aligned with xyz then the north celestial pole (NCP) points directly along
//	the positive z-axis, the point defined by (RA 0, dec 0) points directly along the positive x-axis, and the point defined by
//	(RA 90, dec 0) points directly along the y-axis.
//
//	All rotation matrices used in this code are left-hand. i.e. for an initial uvw vector x, a rotated uvw vector x' and a rotation
//	matrix M, we use x' = Mx. All rotations are anti-clockwise when viewed from a direction such that the normal 2d cartesian axes X
//	and Y are formed by XY, XZ or YZ respectively. To rotate from a UVW coordinate system to world coordinates we first do a latitude
//	rotation about the x-axis by (90 - latitude) so that the phase position moves from (0,0,1) to some point at RA -90. We then do a
//	longitude rotation about the z-axis by (90 + longitude) so that the phase position moves to the required celestial coordinates.
//
//	Parameters:
//
//		1	u (double)
//		2	v (double)
//		3	w (double)
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>

//
//	STRUCTURES
//

// latitude and logitude coordinates.
struct PolarCoords
{
	double longitude;
	double latitude;
	char epoch[20];
};
typedef struct PolarCoords PolarCoords;

// 3-element vector.
struct Vector
{
	double x;
	double y;
	double z;
};
typedef struct Vector Vector;
	
// 3x3 matrix.
struct Matrix
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
typedef struct Matrix Matrix;

//
//	CONSTANTS
//

const double PI = 3.14159265359;
	
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
const char IN_LONG[] = "in_long:";
const char IN_LAT[] = "in_lat:";
const char IN_EPOCH[] = "in_epoch:";
const char OUT_LONG[] = "out_long:";
const char OUT_LAT[] = "out_lat:";
const char OUT_EPOCH[] = "out_epoch:";
const char UV_PROJECTION[] = "uv_projection:";

//
//	GLOBAL VARIABLES
//

// input and output phase centres.
PolarCoords _inPosition;
PolarCoords _outPosition;

// initial uvw coordinates (relative to in phase centre).
Vector _uvw;

// are we doing full uv reprojection, or just calculating the phase change?
bool _uvProjection = false;

// the full rotation matrix from the input to the output coordinate system. this matrix would need
// to be calculated once and then stored so that it can be applied to many baseline vectors.
Matrix _uvwRotation;

// rotation matrix for epoch conversion.
Matrix _epochConversion = { .a11 = 1, .a12 = 0, .a13 = 0, .a21 = 0, .a22 = 1, .a23 = 0, .a31 = 0, .a32 = 0, .a33 = 1 };

//
//	GENERAL FUNCTIONS
//

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

//
//	MATRIX FUNCTIONS
//

//
//	multMatrixVector()
//
//	CJS: 27/07/2015
//
//	Multiply a matrix by a vector.
//

Vector multMatrixVector( Matrix pMatrix, Vector pVector )
{
	
	Vector newVector;
	
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

Matrix multMatrix( Matrix pMatrix1, Matrix pMatrix2 )
{
	
	Matrix newMatrix;
	
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
//	transpose()
//
//	CJS: 28/07/2015
//
//	Construct the transpose of a 3x3 matrix.
//

Matrix transpose( Matrix pOldMatrix )
{
	
	Matrix newMatrix = pOldMatrix;
	
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
//	inverse2x2()
//
//	CJS: 31/07/2015
//
//	Calculate the inverse of a 2x2 matrix. Note that we still use the 3x3 matrix structure, but only refer to the top-left 2x2 cells.
//

Matrix inverse2x2( Matrix pOldMatrix )
{
	
	Matrix newMatrix = pOldMatrix;
	
	double determinant = (pOldMatrix.a11 * pOldMatrix.a22) - (pOldMatrix.a12 * pOldMatrix.a21);
	
	newMatrix.a11 = pOldMatrix.a22 / determinant;
	newMatrix.a12 = -pOldMatrix.a12 / determinant;
	newMatrix.a21 = -pOldMatrix.a21 / determinant;
	newMatrix.a22 = pOldMatrix.a11 / determinant;
	
	// return something.
	return newMatrix;
	
} // inverse2x2

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

Matrix rotateX( double pAngle )
{
	
	Matrix rotationMatrix;
	
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

Matrix rotateY( double pAngle )
{
	
	Matrix rotationMatrix;
	
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

Matrix rotateZ( double pAngle )
{
	
	Matrix rotationMatrix;
	
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
//	convertXYZtoUVW
//
//	CJS: 29/07/2015
//
//	Constructs a rotation matrix that converts coordinates from world coordinates (X = RA 0, dec 0, Y = RA 90, dec 0, Z = RA 0,
//	dec 90) into UVW coordinates (X = U, Y = V, Z = W).
//

Matrix convertXYZtoUVW( PolarCoords pCoords )
{
	
	// rotate the coordinate system so that the pointing direction is at RA -90 deg.
	Matrix rotationMatrix = rotateZ( -(90 + pCoords.longitude) );
	
	// now rotate the coordinate system so that the pointing direction is at the north celestial pole.
	return multMatrix( rotateX( -(90 - pCoords.latitude) ), rotationMatrix );
	
} // convertXYZtoUVW

// 
//	convertUVWtoXYZ
//
//	CJS: 29/07/2015
//
//	Constructs a rotation matrix that converts coordinates from UVW coordinates (X = U, Y = V, Z = W) into world coordinates
//	(X = RA 0, dec 0, Y = RA 90, dec 0, Z = RA 0, dec 90).
//

Matrix convertUVWtoXYZ( PolarCoords pCoords )
{
	
	// rotate the coordinate system so that the pointing direction is at the required dec, and somewhere along RA -90 deg.
	Matrix rotationMatrix = rotateX( 90 - pCoords.latitude );
	
	// row rotate the coordinate system so that the pointing direction is at the required RA as well.
	return multMatrix( rotateZ( pCoords.longitude + 90 ), rotationMatrix );
	
} // convertUVWtoXYZ

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

Matrix epochConversionMatrix( double pNP_RA, double pNP_DEC, double pNP_RA_OFFSET )
{
	
	// rotate about the Z-axis by RA to bring the output north pole to RA zero.
	Matrix rotationMatrix = rotateZ( -pNP_RA );
	
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
//	Construct a matrix that does epoch conversion between two positions. We simply compare the input and output
//	epoch, and then construct a suitable rotation matrix.
//

Matrix doEpochConversion( PolarCoords pIn, PolarCoords pOut )
{
	
	const char J2000[20] = "J2000";
	const char B1950[20] = "B1950";
	const char GALACTIC[20] = "GALACTIC";
	
	// default to no epoch conversion.
	Matrix epochConversion = { .a11 = 1, .a12 = 0, .a13 = 0, .a21 = 0, .a22 = 1, .a23 = 0, .a31 = 0, .a32 = 0, .a33 = 1 };
	
	// J2000 to/from galactic.
	if ((strcmp( pIn.epoch, J2000 ) == 0) && (strcmp( pOut.epoch, GALACTIC ) == 0))
		epochConversion = epochConversionMatrix( NP_RA_GAL_IN_J2000, NP_DEC_GAL_IN_J2000, NP_RA_OFFSET_GAL_IN_J2000 );
	if ((strcmp( pIn.epoch, GALACTIC ) == 0) && (strcmp( pOut.epoch, J2000 ) == 0))
		epochConversion = epochConversionMatrix( NP_RA_J2000_IN_GAL, NP_DEC_J2000_IN_GAL, NP_RA_OFFSET_J2000_IN_GAL );
	
	// B1950 to/from galactic.
	if ((strcmp( pIn.epoch, B1950 ) == 0) && (strcmp( pOut.epoch, GALACTIC ) == 0))
		epochConversion = epochConversionMatrix( NP_RA_GAL_IN_B1950, NP_DEC_GAL_IN_B1950, NP_RA_OFFSET_GAL_IN_B1950 );
	if ((strcmp( pIn.epoch, GALACTIC ) == 0) && (strcmp( pOut.epoch, B1950 ) == 0))
		epochConversion = epochConversionMatrix( NP_RA_B1950_IN_GAL, NP_DEC_B1950_IN_GAL, NP_RA_OFFSET_B1950_IN_GAL );
	
	// B1950 to/from J2000.
	if ((strcmp( pIn.epoch, B1950 ) == 0) && (strcmp( pOut.epoch, J2000 ) == 0))
		epochConversion = epochConversionMatrix( NP_RA_J2000_IN_B1950, NP_DEC_J2000_IN_B1950, NP_RA_OFFSET_J2000_IN_B1950 );
	if ((strcmp( pIn.epoch, J2000 ) == 0) && (strcmp( pOut.epoch, B1950 ) == 0))
		epochConversion = epochConversionMatrix( NP_RA_B1950_IN_J2000, NP_DEC_B1950_IN_J2000, NP_RA_OFFSET_B1950_IN_J2000 );
	
	// return something.
	return epochConversion;
	
} // doEpochConversion

//
//	PHASE-CORRECTION FUNCTIONS
//

//
//	getParameters()
//
//	CJS: 24/07/2015
//
//	Load the parameters from the parameter file 'uvfacetting-params'.
//

void getParameters()
{
	
	const char Y_CONST[] = "Y";
	const char y_CONST[] = "y";

	char params[80], line[1024], par[80];
 
	// Open the parameter file and get all lines.
	FILE *fr = fopen( "uvfacetting-params", "rt" );
	while ( fgets(line, 80, fr) != NULL )
	{

		sscanf( line, "%s %s", par, params );
		if ( strcmp( par, IN_LONG ) == 0)
			_inPosition.longitude = atof( params );
		else if ( strcmp( par, IN_LAT ) == 0 )
			_inPosition.latitude = atof( params );
		else if ( strcmp( par, IN_EPOCH ) == 0 )
		{
			toUppercase( params );
			strcpy( _inPosition.epoch, params );
		}
		else if ( strcmp( par, OUT_LONG ) == 0)
			_outPosition.longitude = atof( params );
		else if ( strcmp( par, OUT_LAT ) == 0 )
			_outPosition.latitude = atof( params );
		else if ( strcmp( par, OUT_EPOCH ) == 0 )
		{
			toUppercase( params );
			strcpy( _outPosition.epoch, params );
		}
		else if ( strcmp( par, UV_PROJECTION ) == 0 )
			_uvProjection = (strcmp( params, Y_CONST ) == 0) || (strcmp( params, y_CONST ) == 0);
            
	}
	fclose(fr);

} // getParameters

//
//	getPathLengthDifference()
//
//	CJS: 24/07/2015
//
//	Rotate the uvw coordinates from the input to the output coordinate system, and determine the change in the path length
//	in order that the visibility phase delay can be calculated.
//

double getPathLengthDifference( Vector pUVW )
{
	
	// The uvw coordinates we've received are relative to the in-phasecentre (i.e. are on a coordinate system where
	// the in-phasecentre is aligned with the Z-axis, so that u == X, v == Y, and w == Z).
	// We must convert the input position into the output coordinate system and epoch.
	Vector inUVW = { .x = 0, .y = 0, .z = 1 };
	Vector phaseI = multMatrixVector( _uvwRotation, inUVW );
	
	// leave the output vector in its own coordinate system and epoch.
	Vector phaseO = { .x = 0, .y = 0, .z = 1 };
	
	// get the vector connecting I to O in the output coordinate system and epoch.
	Vector changeInPosition = { .x = phaseO.x - phaseI.x, .y = phaseO.y - phaseI.y, .z = phaseO.z - phaseI.z };
	
	printf( "\nIn-phasecentre (output C/S): %f %f %f [m]\n", phaseI.x, phaseI.y, phaseI.z );
	printf( "Out-phasecentre (output C/S): %f %f %f [m]\n", phaseO.x, phaseO.y, phaseO.z );
	printf( "Change in position (output C/S): %f %f %f [m]\n\n", changeInPosition.x, changeInPosition.y, changeInPosition.z );
	
	// the uvw vector is in the input coordinate system, so we need to rotate it so that it is in the output coordinate system and epoch.
	Vector uvwRotated = multMatrixVector( _uvwRotation, pUVW );
	
	// now the path length difference is given by the dot product of our newly constructed vector with uvw.
	return (changeInPosition.x * uvwRotated.x) + (changeInPosition.y * uvwRotated.y) + (changeInPosition.z * uvwRotated.z);
	
} // getPathLengthDifference

//
//	reprojectUV
//
//	CJS: 24/07/2015
//
//	Do a full uv-reprojection on the UVW coordinates. This code reproduces the equations shown in Appendix A of
//	Sault et al 1996 A&ASS 120 375-384. In the image domain we require a rotational transform R from the input to the output
//	coordinate system, and in the UV-domain the required matrix is R^-1[T] (eqn A4).
//

Matrix reprojectUV()
{
	
	// from Sault we have:
	//
	//	(u,v)' = M x (u,v)
	//
	// where	M = 	(a11 a12) / n
	//			(a21 a22)
	// and		a11 = cos(ra_in - ra_out)sin(dec_in)sin(dec_out) + cos(dec_in)cos(dec_out)
	//		a12 = -sin(ra_in - ra_out)sin(dec_out)
	//		a21 = cos(ra_in - ra_out)
	//		a22 = cos(ra_in - ra_out)
	//		n = sin(dec_in)sin(dec_out) + cos(ra_in - ra_out)cos(dec_in)cos(dec_out)
	//
	// these values are found in the matrix _uvwRotation, and we could just pick them from here. Note that we've rename the Aij indexing
	// of the matrix cells to use the more conventional a_row_column rather than the a_column_row given in Sault.
	//
	// however, the proper way to calculate the 2x2 matrix M is to take the top left 2x2 cells of _uvwRotation, take the inverse, and
	// then transpose the result.
	
	// extract the top-left 2x2 matrix from _uvwRotation.
	Matrix rotationMatrixReduced = {	.a11 = _uvwRotation.a11, .a12 = _uvwRotation.a12, .a13 = 0,
						.a21 = _uvwRotation.a21, .a22 = _uvwRotation.a22, .a23 = 0,
						.a31 = 0, .a32 = 0, .a33 = 1 };
						
	// get the 2x2 inverse and then the 2x2 transpose.
	rotationMatrixReduced = transpose( inverse2x2( rotationMatrixReduced ) );
	
	// display reduced matrix for debugging.
	printf( "rot4: 11 (%f), 12 (%f), 21 (%f), 22 (%f)\n\n", rotationMatrixReduced.a11, rotationMatrixReduced.a12,
								rotationMatrixReduced.a21, rotationMatrixReduced.a22 );
	
	// return something.
	return rotationMatrixReduced;
	
} // reprojectUV

//
//	main()
//
//	CJS: 24/07/2015
//
//	Load the parameters from the parameter file 'uvfacetting-params'.
//	Get the path length difference.
//

int main( int pArgc, char ** pArgv )
{
	
	// read program arguments. we expect to see the program call (0) and the uvw coordinates (1-3). the coordinates
	// should be specified in metres.
	if (pArgc != 4)
	{
		printf("Wrong number of arguments. Require u, v and w coordinates.\n");
		return 1;
	}
	_uvw.x = atof( pArgv[1] );
	_uvw.y = atof( pArgv[2] );
	_uvw.z = atof( pArgv[3] );

	// get the run-time parameters, which specify the in and out phase centres.
	getParameters();
	
	// prepare the epoch conversion matrix.
	_epochConversion = doEpochConversion( _inPosition, _outPosition );

	// calculate the uvw rotation matrix that transforms the uvw coordinates relative to the in phase centre to uvw
	// coordinates that are relative to the out phase centre. we use latitude and longitude rotations about
	// both the in and out phase positions, and include an epoch-conversion step in between.
	_uvwRotation = convertUVWtoXYZ( _inPosition );
	
	// now do epoch conversion.
	_uvwRotation = multMatrix( _epochConversion, _uvwRotation );
	
	// we probably want to see the current coordinates here for debugging purposes, but this code can be removed.
	Vector tmpUVW = multMatrixVector( _uvwRotation, _uvw );
	printf("--> uvw in output epoch (world coordinates): %f %f %f\n", tmpUVW.x, tmpUVW.y, tmpUVW.z);
	
	// finally, rotate into UVW coordinates relative to the output position.
	_uvwRotation = multMatrix( convertXYZtoUVW( _outPosition ), _uvwRotation );
	
	printf( "\nuvrot matrix:\n\n"
		"    %f %f %f\n"
		"    %f %f %f\n"
		"    %f %f %f\n",	_uvwRotation.a11, _uvwRotation.a12,_uvwRotation.a13,
					_uvwRotation.a21, _uvwRotation.a22, _uvwRotation.a23,
					_uvwRotation.a31, _uvwRotation.a32, _uvwRotation.a33 );
		
	// get the additional path length difference when switching from the in to the out phase centre..
	double pathLengthDifference = getPathLengthDifference( _uvw );
		
	printf( "additional path length difference: %f m\n\n", pathLengthDifference );
		
	// are we are doing full uv reprojection? If not, then just convert to output coordinates.
	Vector newUVW = multMatrixVector( _uvwRotation, _uvw );
	if (_uvProjection == true)
		newUVW = multMatrixVector( reprojectUV(), newUVW );
	
	printf( "New uvw in output coordinates:\n\n    %f %f %f [m]\n\n", newUVW.x, newUVW.y, newUVW.z );
	
	// we now have the re-projected uv coordinates, but they are in output coordinates. we want them back in input coordinates. We
	// can use the transpose of the rotation matrix to do this, because the transpose of a rotation matrix is the same as
	// the inverse.
	Vector newVector = multMatrixVector( transpose( _uvwRotation ), newUVW );
	
	printf( "New uvw converted back from output to input coordinates:\n\n    %f %f %f [m]\n\n", newVector.x, newVector.y, newVector.z );

	return 0;
  
} // main
