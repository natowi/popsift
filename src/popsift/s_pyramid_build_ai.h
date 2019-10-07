#include "hip/hip_runtime.h"
/*
 * Copyright 2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/plane_2d.h"

namespace popsift {
namespace gauss {
namespace absoluteSourceInterpolated {

__global__
void horiz( hipTextureObject_t src_linear_tex,
            hipSurfaceObject_t dst_data,
            const int           dst_level );

__global__
void vert( hipTextureObject_t src_linear_tex,
           hipSurfaceObject_t dst_data,
           const int           dst_level );

__global__
void vert_abs0( hipTextureObject_t src_linear_tex,
           hipSurfaceObject_t dst_data,
           const int           dst_level );

__global__
void vert_all_abs0( hipTextureObject_t src_linear_tex,
                    hipSurfaceObject_t dst_data,
                    const int           start_level,
                    const int           max_level );

} // namespace absoluteSourceInterpolated
} // namespace gauss
} // namespace popsift

