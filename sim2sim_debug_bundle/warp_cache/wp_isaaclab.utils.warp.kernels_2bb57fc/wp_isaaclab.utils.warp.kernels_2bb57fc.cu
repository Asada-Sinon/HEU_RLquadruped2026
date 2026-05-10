
#define WP_TILE_BLOCK_DIM 256
#define WP_NO_CRT
#include "builtin.h"

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

#define builtin_tid1d() wp::tid(_idx, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, _idx, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, _idx, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, _idx, dim)

#define builtin_block_dim() wp::block_dim()

extern "C" {
}


extern "C" __global__ void reshape_tiled_image_5f4bc01e_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::uint32> var_tiled_image_buffer,
    wp::array_t<wp::uint32> var_batched_image,
    wp::int32 var_image_height,
    wp::int32 var_image_width,
    wp::int32 var_num_channels,
    wp::int32 var_num_tiles_x)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::range_t var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::uint32* var_18;
        wp::uint32 var_19;
        wp::uint32 var_20;
        //---------
        // forward
        // def reshape_tiled_image(                                                               <L 1>
        // camera_id, height_id, width_id = wp.tid()                                              <L 24>
        builtin_tid3d(var_0, var_1, var_2);
        // tile_x_id = camera_id % num_tiles_x                                                    <L 27>
        var_3 = wp::mod(var_0, var_num_tiles_x);
        // tile_y_id = camera_id // num_tiles_x                                                   <L 28>
        var_4 = wp::floordiv(var_0, var_num_tiles_x);
        // pixel_start = (                                                                        <L 30>
        // num_channels * num_tiles_x * image_width * (image_height * tile_y_id + height_id)       <L 31>
        var_5 = wp::mul(var_num_channels, var_num_tiles_x);
        var_6 = wp::mul(var_5, var_image_width);
        var_7 = wp::mul(var_image_height, var_4);
        var_8 = wp::add(var_7, var_1);
        var_9 = wp::mul(var_6, var_8);
        // + num_channels * tile_x_id * image_width                                               <L 32>
        var_10 = wp::mul(var_num_channels, var_3);
        var_11 = wp::mul(var_10, var_image_width);
        var_12 = wp::add(var_9, var_11);
        // + num_channels * width_id                                                              <L 33>
        var_13 = wp::mul(var_num_channels, var_2);
        var_14 = wp::add(var_12, var_13);
        // for i in range(num_channels):                                                          <L 37>
        var_15 = wp::range(var_num_channels);
        start_for_0:;
            if (iter_cmp(var_15) == 0) goto end_for_0;
            var_16 = wp::iter_next(var_15);
            // batched_image[camera_id, height_id, width_id, i] = batched_image.dtype(tiled_image_buffer[pixel_start + i])       <L 38>
            var_17 = wp::add(var_14, var_16);
            var_18 = wp::address(var_tiled_image_buffer, var_17);
            var_20 = wp::load(var_18);
            var_19 = wp::uint32(var_20);
            wp::array_store(var_batched_image, var_0, var_1, var_2, var_16, var_19);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void reshape_tiled_image_43f9491f_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::uint8> var_tiled_image_buffer,
    wp::array_t<wp::uint8> var_batched_image,
    wp::int32 var_image_height,
    wp::int32 var_image_width,
    wp::int32 var_num_channels,
    wp::int32 var_num_tiles_x)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::range_t var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::uint8* var_18;
        wp::uint8 var_19;
        wp::uint8 var_20;
        //---------
        // forward
        // def reshape_tiled_image(                                                               <L 1>
        // camera_id, height_id, width_id = wp.tid()                                              <L 24>
        builtin_tid3d(var_0, var_1, var_2);
        // tile_x_id = camera_id % num_tiles_x                                                    <L 27>
        var_3 = wp::mod(var_0, var_num_tiles_x);
        // tile_y_id = camera_id // num_tiles_x                                                   <L 28>
        var_4 = wp::floordiv(var_0, var_num_tiles_x);
        // pixel_start = (                                                                        <L 30>
        // num_channels * num_tiles_x * image_width * (image_height * tile_y_id + height_id)       <L 31>
        var_5 = wp::mul(var_num_channels, var_num_tiles_x);
        var_6 = wp::mul(var_5, var_image_width);
        var_7 = wp::mul(var_image_height, var_4);
        var_8 = wp::add(var_7, var_1);
        var_9 = wp::mul(var_6, var_8);
        // + num_channels * tile_x_id * image_width                                               <L 32>
        var_10 = wp::mul(var_num_channels, var_3);
        var_11 = wp::mul(var_10, var_image_width);
        var_12 = wp::add(var_9, var_11);
        // + num_channels * width_id                                                              <L 33>
        var_13 = wp::mul(var_num_channels, var_2);
        var_14 = wp::add(var_12, var_13);
        // for i in range(num_channels):                                                          <L 37>
        var_15 = wp::range(var_num_channels);
        start_for_0:;
            if (iter_cmp(var_15) == 0) goto end_for_0;
            var_16 = wp::iter_next(var_15);
            // batched_image[camera_id, height_id, width_id, i] = batched_image.dtype(tiled_image_buffer[pixel_start + i])       <L 38>
            var_17 = wp::add(var_14, var_16);
            var_18 = wp::address(var_tiled_image_buffer, var_17);
            var_20 = wp::load(var_18);
            var_19 = wp::uint8(var_20);
            wp::array_store(var_batched_image, var_0, var_1, var_2, var_16, var_19);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void reshape_tiled_image_45364ab5_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_tiled_image_buffer,
    wp::array_t<wp::float32> var_batched_image,
    wp::int32 var_image_height,
    wp::int32 var_image_width,
    wp::int32 var_num_channels,
    wp::int32 var_num_tiles_x)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::range_t var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        //---------
        // forward
        // def reshape_tiled_image(                                                               <L 1>
        // camera_id, height_id, width_id = wp.tid()                                              <L 24>
        builtin_tid3d(var_0, var_1, var_2);
        // tile_x_id = camera_id % num_tiles_x                                                    <L 27>
        var_3 = wp::mod(var_0, var_num_tiles_x);
        // tile_y_id = camera_id // num_tiles_x                                                   <L 28>
        var_4 = wp::floordiv(var_0, var_num_tiles_x);
        // pixel_start = (                                                                        <L 30>
        // num_channels * num_tiles_x * image_width * (image_height * tile_y_id + height_id)       <L 31>
        var_5 = wp::mul(var_num_channels, var_num_tiles_x);
        var_6 = wp::mul(var_5, var_image_width);
        var_7 = wp::mul(var_image_height, var_4);
        var_8 = wp::add(var_7, var_1);
        var_9 = wp::mul(var_6, var_8);
        // + num_channels * tile_x_id * image_width                                               <L 32>
        var_10 = wp::mul(var_num_channels, var_3);
        var_11 = wp::mul(var_10, var_image_width);
        var_12 = wp::add(var_9, var_11);
        // + num_channels * width_id                                                              <L 33>
        var_13 = wp::mul(var_num_channels, var_2);
        var_14 = wp::add(var_12, var_13);
        // for i in range(num_channels):                                                          <L 37>
        var_15 = wp::range(var_num_channels);
        start_for_0:;
            if (iter_cmp(var_15) == 0) goto end_for_0;
            var_16 = wp::iter_next(var_15);
            // batched_image[camera_id, height_id, width_id, i] = batched_image.dtype(tiled_image_buffer[pixel_start + i])       <L 38>
            var_17 = wp::add(var_14, var_16);
            var_18 = wp::address(var_tiled_image_buffer, var_17);
            var_20 = wp::load(var_18);
            var_19 = wp::float32(var_20);
            wp::array_store(var_batched_image, var_0, var_1, var_2, var_16, var_19);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void raycast_mesh_kernel_2b491393_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::uint64 var_mesh,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_starts,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_directions,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_hits,
    wp::array_t<wp::float32> var_ray_distance,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_normal,
    wp::array_t<wp::int32> var_ray_face_id,
    wp::float32 var_max_dist,
    wp::int32 var_return_distance,
    wp::int32 var_return_normal,
    wp::int32 var_return_face_id)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::float32 var_1 = 0.0;
        wp::float32 var_2;
        const wp::float32 var_3 = 0.0;
        wp::float32 var_4;
        const wp::float32 var_5 = 0.0;
        wp::float32 var_6;
        const wp::float32 var_7 = 0.0;
        wp::float32 var_8;
        wp::vec_t<3, wp::float32> var_9;
        const wp::int32 var_10 = 0;
        wp::int32 var_11;
        wp::vec_t<3, wp::float32>* var_12;
        wp::vec_t<3, wp::float32>* var_13;
        bool var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32>* var_17;
        wp::vec_t<3, wp::float32>* var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32> var_22;
        const wp::int32 var_23 = 1;
        bool var_24;
        const wp::int32 var_25 = 1;
        bool var_26;
        const wp::int32 var_27 = 1;
        bool var_28;
        //---------
        // forward
        // def raycast_mesh_kernel(                                                               <L 14>
        // tid = wp.tid()                                                                         <L 56>
        var_0 = builtin_tid1d();
        // t = float(0.0)  # hit distance along ray                                               <L 58>
        var_2 = wp::float(var_1);
        // u = float(0.0)  # hit face barycentric u                                               <L 59>
        var_4 = wp::float(var_3);
        // v = float(0.0)  # hit face barycentric v                                               <L 60>
        var_6 = wp::float(var_5);
        // sign = float(0.0)  # hit face sign                                                     <L 61>
        var_8 = wp::float(var_7);
        // n = wp.vec3()  # hit face normal                                                       <L 62>
        var_9 = wp::vec_t<3, wp::float32>();
        // f = int(0)  # hit face index                                                           <L 63>
        var_11 = wp::int(var_10);
        // hit_success = wp.mesh_query_ray(mesh, ray_starts[tid], ray_directions[tid], max_dist, t, u, v, sign, n, f)       <L 66>
        var_12 = wp::address(var_ray_starts, var_0);
        var_13 = wp::address(var_ray_directions, var_0);
        var_15 = wp::load(var_12);
        var_16 = wp::load(var_13);
        var_14 = wp::mesh_query_ray(var_mesh, var_15, var_16, var_max_dist, var_2, var_4, var_6, var_8, var_9, var_11);
        // if hit_success:                                                                        <L 68>
        if (var_14) {
            // ray_hits[tid] = ray_starts[tid] + t * ray_directions[tid]                          <L 69>
            var_17 = wp::address(var_ray_starts, var_0);
            var_18 = wp::address(var_ray_directions, var_0);
            var_20 = wp::load(var_18);
            var_19 = wp::mul(var_2, var_20);
            var_22 = wp::load(var_17);
            var_21 = wp::add(var_22, var_19);
            wp::array_store(var_ray_hits, var_0, var_21);
            // if return_distance == 1:                                                           <L 70>
            var_24 = (var_return_distance == var_23);
            if (var_24) {
                // ray_distance[tid] = t                                                          <L 71>
                wp::array_store(var_ray_distance, var_0, var_2);
            }
            // if return_normal == 1:                                                             <L 72>
            var_26 = (var_return_normal == var_25);
            if (var_26) {
                // ray_normal[tid] = n                                                            <L 73>
                wp::array_store(var_ray_normal, var_0, var_9);
            }
            // if return_face_id == 1:                                                            <L 74>
            var_28 = (var_return_face_id == var_27);
            if (var_28) {
                // ray_face_id[tid] = f                                                           <L 75>
                wp::array_store(var_ray_face_id, var_0, var_11);
            }
        }
    }
}

