//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Purpose: Native AltiVec (VMX) math primitives for PowerPC.
//
// This file is only compiled on PowerPC with AltiVec enabled (see
// mathlib/wscript).  It supplies PPC-native implementations of the
// _SSE_* scalar routines that sse.cpp otherwise provides for x86/ARM;
// on PowerPC+AltiVec builds sse.cpp excludes those definitions (guarded
// with #if !defined(__powerpc__) || !defined(__ALTIVEC__)) so there is
// no duplicate-symbol conflict and the x86_64 build path is untouched.
//
// Only base AltiVec instructions are used (G4/G5/970, POWER6-class):
// no VSX, no VMX128, no VMX256.  Safe for the PowerPC G5 built with
// -mcpu=power4 -maltivec.
//=====================================================================================//

#include <math.h>
#include <float.h>	// Needed for FLT_EPSILON / FLT_MIN
#include "basetypes.h"
#include <memory.h>
#include "tier0/dbg.h"
#include "mathlib/mathlib.h"
#include "mathlib/vector.h"
#include "sse.h"

// memdbgon must be the last include file in a .cpp file!!!
#include "tier0/memdbgon.h"

#if defined(__powerpc__) && defined(__ALTIVEC__)
#include <altivec.h>

// Reciprocal-square-root estimate with a floor to keep the estimate finite.
static inline __vector float AltiVec_RSqrtEstimate( __vector float v )
{
	__vector float minVal = { FLT_MIN, FLT_MIN, FLT_MIN, FLT_MIN };
	return vec_rsqrte( vec_max( v, minVal ) );
}

// One Newton-Raphson refinement of a reciprocal-square-root estimate:
//	est' = 0.5 * est * (3 - a * est^2)
static inline __vector float AltiVec_RSqrtRefine( __vector float a, __vector float est )
{
	__vector float zero  = { 0.f, 0.f, 0.f, 0.f };
	__vector float three = { 3.f, 3.f, 3.f, 3.f };
	__vector float half  = { 0.5f, 0.5f, 0.5f, 0.5f };
	__vector float estSq = vec_madd( est, est, zero );
	__vector float refine = vec_nmsub( a, estSq, three );
	__vector float halfEst = vec_madd( est, half, zero );
	return vec_madd( halfEst, refine, est );
}

// AltiVec has no hardware vector square root: sqrt(x) = x * rsqrt(x)
float _SSE_Sqrt(float x)
{
	Assert( s_bMathlibInitialized );
	return x * _SSE_RSqrtAccurate( x );
}

// Single iteration NewtonRaphson reciprocal square root:
// 0.5 * rsqrt(x) * (3 - x * rsqrt(x)^2)
float _SSE_RSqrtAccurate(float a)
{
	Assert( s_bMathlibInitialized );

	__vector float v = { a, a, a, a };
	__vector float est = AltiVec_RSqrtEstimate( v );
	__vector float result = AltiVec_RSqrtRefine( v, est );
	return vec_extract( result, 0 );
}

// Simple AltiVec rsqrt estimate.  Usually accurate to around 6 (relative)
// decimal places or so, so ok for closed transforms. (ie, computing lighting
// normals)
float _SSE_RSqrtFast(float x)
{
	Assert( s_bMathlibInitialized );

	__vector float v = { x, x, x, x };
	return vec_extract( AltiVec_RSqrtEstimate( v ), 0 );
}

#endif // __powerpc__ && __ALTIVEC__
