/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <iostream>
#include <vector>

#include "s_image.h"
#include "sift_conf.h"
#include "sift_extremum.h"
#include "sift_constants.h"

namespace popsift {

struct LinearTexture
{
    hipSurfaceObject_t tex;
};

class Octave
{
    int   _w;
    int   _h;
    int   _max_w;
    int   _max_h;
    float _w_grid_divider;
    float _h_grid_divider;
    int   _debug_octave_id;
    int   _levels;
    int   _gauss_group;

    hipArray_t           _data;
    hipChannelFormatDesc _data_desc;
    hipExtent            _data_ext;
    hipSurfaceObject_t   _data_surf;
    hipTextureObject_t   _data_tex_point;
    LinearTexture         _data_tex_linear;

    hipArray_t           _intm;
    hipChannelFormatDesc _intm_desc;
    hipExtent            _intm_ext;
    hipSurfaceObject_t   _intm_surf;
    hipTextureObject_t   _intm_tex_point;
    LinearTexture         _intm_tex_linear;

    hipArray_t           _dog_3d;
    hipChannelFormatDesc _dog_3d_desc;
    hipExtent            _dog_3d_ext;
    hipSurfaceObject_t   _dog_3d_surf;
    hipTextureObject_t   _dog_3d_tex_point;
    LinearTexture         _dog_3d_tex_linear;

    // one CUDA stream per level
    // consider whether some of them can be removed
    hipStream_t _stream;
    hipEvent_t  _scale_done;
    hipEvent_t  _extrema_done;
    hipEvent_t  _ori_done;
    hipEvent_t  _desc_done;

public:
    Octave( );
    ~Octave( ) { this->free(); }

    void resetDimensions( const Config& conf, int w, int h );

    inline void debugSetOctave( uint32_t o ) { _debug_octave_id = o; }

    inline int getLevels() const { return _levels; }
    inline int getWidth() const  {
        return _w;
    }
    inline int getHeight() const {
        return _h;
    }

    inline float getWGridDivider() const  {
        return _w_grid_divider;
    }
    inline float getHGridDivider() const {
        return _h_grid_divider;
    }

    inline hipStream_t getStream( ) {
        return _stream;
    }
    inline hipEvent_t getEventScaleDone( ) {
        return _scale_done;
    }
    inline hipEvent_t getEventExtremaDone( ) {
        return _extrema_done;
    }
    inline hipEvent_t getEventOriDone( ) {
        return _ori_done;
    }
    inline hipEvent_t getEventDescDone( ) {
        return _desc_done;
    }

    inline LinearTexture getIntermDataTexLinear( ) {
        return _intm_tex_linear;
    }
    inline hipTextureObject_t getIntermDataTexPoint( ) {
        return _intm_tex_point;
    }
    inline LinearTexture getDataTexLinear( ) {
        return _data_tex_linear;
    }
    inline hipTextureObject_t getDataTexPoint( ) {
        return _data_tex_point;
    }
    inline hipSurfaceObject_t getDataSurface( ) {
        return _data_surf;
    }
    inline hipSurfaceObject_t getIntermediateSurface( ) {
        return _intm_surf;
    }
        
    inline hipSurfaceObject_t& getDogSurface( ) {
        return _dog_3d_surf;
    }
    inline hipTextureObject_t& getDogTexturePoint( ) {
        return _dog_3d_tex_point;
    }
    inline LinearTexture& getDogTextureLinear( ) {
        return _dog_3d_tex_linear;
    }

    /**
     * alloc() - allocates all GPU memories for one octave
     * @param width in floats, not bytes!!!
     */
    void alloc( const Config& conf,
                int           width,
                int           height,
                int           levels,
                int           gauss_group );
    void free();

    /**
     * debug:
     * download a level and write to disk
     */
    void download_and_save_array( const char* basename, int octave );

private:
    void alloc_data_planes( );
    void alloc_data_tex( );
    void alloc_interm_array( );
    void alloc_interm_tex( );
    void alloc_dog_array( );
    void alloc_dog_tex( );
    void alloc_streams( );
    void alloc_events( );

    void free_events( );
    void free_streams( );
    void free_dog_tex( );
    void free_dog_array( );
    void free_interm_tex( );
    void free_interm_array( );
    void free_data_tex( );
    void free_data_planes( );
};

} // namespace popsift
