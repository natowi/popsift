/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <stdlib.h>
#include <assert.h>
#include <hip/hip_runtime.h>

// synchronize device and check for an error
void pop_sync_check_last_error( const char* file, size_t line );

// check for an error without synchronizing first
void pop_check_last_error( const char* file, size_t      line );

#define POP_CHK pop_check_last_error( __FILE__, __LINE__ )

#ifdef ERRCHK_AFTER_KERNEL
#define POP_SYNC_CHK pop_sync_check_last_error( __FILE__, __LINE__ )
#else
#define POP_SYNC_CHK
#endif

namespace popsift {
namespace cuda {
void malloc_dev( void** ptr, int sz,
                 const char* file, int line );

void malloc_hst( void** ptr, int sz,
                 const char* file, int line );

template<class T>
T* malloc_devT( int num, const char* file, int line ) {
    void* ptr;
    malloc_dev( &ptr, num*sizeof(T), file, line );
    return (T*)ptr;
}

template<class T>
T* malloc_hstT( int num, const char* file, int line ) {
    void* ptr;
    malloc_hst( &ptr, num*sizeof(T), file, line );
    return (T*)ptr;
}

void memcpy_sync( void* dst, const void* src, size_t sz,
                   hipMemcpyKind type,
                   const char* file, size_t line );
#define popcuda_memcpy_sync( dst, src, sz, type ) \
    popsift::cuda::memcpy_sync( dst, src, sz, type, __FILE__, __LINE__ )

void memcpy_async( void* dst, const void* src, size_t sz,
                   hipMemcpyKind type, hipStream_t stream,
                   const char* file, size_t line );
#define popcuda_memcpy_async( dst, src, sz, type, stream ) \
    popsift::cuda::memcpy_async( dst, src, sz, type, stream, __FILE__, __LINE__ )

void memset_sync( void* ptr, int value, size_t bytes, const char* file, size_t line );
#define popcuda_memset_sync( ptr, val, sz ) \
    popsift::cuda::memset_sync( ptr, val, sz, __FILE__, __LINE__ )

void memset_async( void* ptr, int value, size_t bytes, hipStream_t stream, const char* file, size_t line );
#define popcuda_memset_async( ptr, val, sz, stream ) \
    popsift::cuda::memset_async( ptr, val, sz, stream, __FILE__, __LINE__ )

hipStream_t stream_create( const char* file, size_t line );
void         stream_destroy( hipStream_t s, const char* file, size_t line );
hipEvent_t  event_create( const char* file, size_t line );
void         event_destroy( hipEvent_t ev, const char* file, size_t line );
void         event_record( hipEvent_t ev, hipStream_t s, const char* file, size_t line );
void         event_wait( hipEvent_t ev, hipStream_t s, const char* file, size_t line );
float        event_diff( hipEvent_t from, hipEvent_t to );

class BriefDuration
{
    hipStream_t _stream;
    hipEvent_t  _on;
    hipEvent_t  _off;
public:
    BriefDuration( hipStream_t s, const char* file, size_t line )
        : _stream( s )
    {
        _on  = event_create( file, line );
        _off = event_create( file, line );
    }

    ~BriefDuration( )
    {
        event_destroy( _on,  __FILE__, __LINE__ );
        event_destroy( _off, __FILE__, __LINE__ );
    }

    void start( const char* file, size_t line ) {
        hipStreamSynchronize( _stream );
        event_record( _on, _stream, file, line );
    }

    void stop( const char* file, size_t line ) {
        event_record( _off, _stream, file, line );
    }

    float report( const char* file, size_t line ) {
        event_wait( _off, _stream, file, line );
        hipStreamSynchronize( _stream );
        return event_diff( _on, _off );
    }
};

};
};

#define POP_FATAL(s) { \
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl << "    " << s << std::endl; \
        exit( -__LINE__ ); \
    }

#define POP_FATAL_FL(s,file,line) { \
        std::cerr << file << ":" << line << std::endl << "    " << s << std::endl; \
        exit( -__LINE__ ); \
    }

#define POP_CHECK_NON_NULL(ptr,s) if( ptr == 0 ) { POP_FATAL_FL(s,__FILE__,__LINE__); }

#define POP_CHECK_NON_NULL_FL(ptr,s,file,line) if( ptr == 0 ) { POP_FATAL_FL(s,file,line); }

#define POP_INFO(s)
// #define POP_INFO(s) cerr << __FILE__ << ":" << __LINE__ << std::endl << "    " << s << endl

#define POP_INFO2(silent,s) \
    if (not silent) { \
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl << "    " << s << std::endl; \
    }

#define POP_CUDA_FATAL(err,s) { \
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl; \
        std::cerr << "    " << s << hipGetErrorString(err) << std::endl; \
        exit( -__LINE__ ); \
    }
#define POP_CUDA_FATAL_TEST(err,s) if( err != hipSuccess ) { POP_CUDA_FATAL(err,s); }

#define POP_CUDA_FREE( ptr ) { \
        hipError_t err; \
        err = hipFree( ptr ); \
        POP_CUDA_FATAL_TEST( err, "hipFree failed: " ); \
    }

#define POP_CUDA_MALLOC_HOST( ptr, sz ) { \
        hipError_t err; \
        err = hipHostMalloc( ptr, sz ); \
        POP_CUDA_FATAL_TEST( err, "hipHostMalloc failed: " ); \
    }

#define POP_CUDA_FREE_HOST( ptr ) { \
        hipError_t err; \
        err = hipHostFree( ptr ); \
        POP_CUDA_FATAL_TEST( err, "hipHostFree failed: " ); \
    }

