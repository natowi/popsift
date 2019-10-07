/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "debug_macros.h"

#include <assert.h>

using namespace std;

void pop_sync_check_last_error( const char* file, size_t line )
{
    hipDeviceSynchronize();
    pop_check_last_error( file, line );
}

void pop_check_last_error( const char* file, size_t line )
{
    hipError_t err = hipGetLastError( );
    if( err != hipSuccess ) {
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl
                  << "    called from " << file << ":" << line << std::endl
                  << "    hipGetLastError failed: " << hipGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
}

namespace popsift { namespace cuda {
void malloc_dev( void** ptr, int sz,
                 const char* file, int line )
{
    hipError_t err;
    err = hipMalloc( ptr, sz );
    if( err != hipSuccess ) {
        std::cerr << file << ":" << line << std::endl
                  << "    hipMalloc failed: " << hipGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
#ifdef DEBUG_INIT_DEVICE_ALLOCATIONS
    popsift::cuda::memset_sync( *ptr, 0, sz, file, line );
#endif // NDEBUG
}
} }

namespace popsift { namespace cuda {
void malloc_hst( void** ptr, int sz,
                 const char* file, int line )
{
    hipError_t err;
    err = hipHostMalloc( ptr, sz );
    if( err != hipSuccess ) {
        std::cerr << file << ":" << line << std::endl
                  << "    hipHostMalloc failed: " << hipGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
#ifdef DEBUG_INIT_DEVICE_ALLOCATIONS
    memset( *ptr, 0, sz );
#endif // NDEBUG
}
} }

namespace popsift { namespace cuda {
void memcpy_async( void* dst, const void* src, size_t sz,
                   hipMemcpyKind type, hipStream_t stream,
                   const char* file, size_t line )
{
    POP_CHECK_NON_NULL_FL( dst, "Dest ptr in memcpy async is null.", file, line );
    POP_CHECK_NON_NULL_FL( src, "Source ptr in memcpy async is null.", file, line );
    POP_CHECK_NON_NULL_FL( sz, "Size in memcpy async is null.", file, line );

    hipError_t err;
    err = hipMemcpyAsync( dst, src, sz, type, stream );
    if( err != hipSuccess ) {
        cerr << file << ":" << line << endl
             << "    " << "Failed to copy "
             << (type==hipMemcpyHostToDevice?"host-to-device":"device-to-host")
             << ": ";
        cerr << hipGetErrorString(err) << endl;
        cerr << "    src ptr=" << hex << (size_t)src << dec << endl
             << "    dst ptr=" << hex << (size_t)dst << dec << endl;
        exit( -__LINE__ );
    }
    POP_CUDA_FATAL_TEST( err, "Failed to copy host-to-device: " );
}

void memcpy_sync( void* dst, const void* src, size_t sz, hipMemcpyKind type, const char* file, size_t line )
{
    POP_CHECK_NON_NULL( dst, "Dest ptr in memcpy async is null." );
    POP_CHECK_NON_NULL( src, "Source ptr in memcpy async is null." );
    POP_CHECK_NON_NULL( sz, "Size in memcpy async is null." );

    hipError_t err;
    err = hipMemcpy( dst, src, sz, type );
    if( err != hipSuccess ) {
        cerr << "    " << "Failed to copy "
             << (type==hipMemcpyHostToDevice?"host-to-device":"device-to-host")
             << ": ";
        cerr << hipGetErrorString(err) << endl;
        cerr << "    src ptr=" << hex << (size_t)src << dec << endl
             << "    dst ptr=" << hex << (size_t)dst << dec << endl;
        exit( -__LINE__ );
    }
    POP_CUDA_FATAL_TEST( err, "Failed to copy host-to-device: " );
}

void memset_async( void* ptr, int value, size_t bytes, hipStream_t stream, const char* file, size_t line )
{
    hipError_t err;
    err = hipMemsetAsync( ptr, value, bytes, stream );
    if( err != hipSuccess ) {
        std::cerr << file << ":" << line << std::endl
                  << "    hipMemsetAsync failed: " << hipGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
}

void memset_sync( void* ptr, int value, size_t bytes, const char* file, size_t line )
{
    hipError_t err;
    err = hipMemset( ptr, value, bytes );
    if( err != hipSuccess ) {
        std::cerr << file << ":" << line << std::endl
                  << "    hipMemset failed: " << hipGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
}
} }

namespace popsift { namespace cuda {
hipStream_t stream_create( const char* file, size_t line )
{
    hipStream_t stream;
    hipError_t err;
    err = hipStreamCreate( &stream );
    if( err != hipSuccess ) {
        std::cerr << file << ":" << line << std::endl
                  << "    hipStreamCreate failed: " << hipGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
    return stream;
}
void stream_destroy( hipStream_t s, const char* file, size_t line )
{
    hipError_t err;
    err = hipStreamDestroy( s );
    if( err != hipSuccess ) {
        std::cerr << file << ":" << line << std::endl
                  << "    hipStreamDestroy failed: " << hipGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
}
hipEvent_t event_create( const char* file, size_t line )
{
    hipEvent_t ev;
    hipError_t err;
    err = hipEventCreate( &ev );
    if( err != hipSuccess ) {
        std::cerr << file << ":" << line << std::endl
                  << "    hipEventCreate failed: " << hipGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
    return ev;
}
void event_destroy( hipEvent_t ev, const char* file, size_t line )
{
    hipError_t err;
    err = hipEventDestroy( ev );
    if( err != hipSuccess ) {
        std::cerr << file << ":" << line << std::endl
                  << "    hipEventDestroy failed: " << hipGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
}
void event_record( hipEvent_t ev, hipStream_t s, const char* file, size_t line )
{
    hipError_t err;
    err = hipEventRecord( ev, s );
    if( err != hipSuccess ) {
        std::cerr << file << ":" << line << std::endl
                  << "    hipEventRecord failed: " << hipGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
}
void event_wait( hipEvent_t ev, hipStream_t s, const char* file, size_t line )
{
    hipError_t err;
    err = hipStreamWaitEvent( s, ev, 0 );
    if( err != hipSuccess ) {
        std::cerr << file << ":" << line << std::endl
                  << "    hipStreamWaitEvent failed: " << hipGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
}

float event_diff( hipEvent_t from, hipEvent_t to )
{   
    float ms;
    hipEventElapsedTime( &ms, from, to );
    return ms;
}

} // namespace cuda
} // namespace popsift

