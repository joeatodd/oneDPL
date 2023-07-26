// -*- C++ -*-
//===-- parallel_backend_sycl_reduce.h --------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_H

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "sycl_defs.h"
#include "parallel_backend_sycl_utils.h"
#include "execution_sycl_defs.h"
#include "unseq_backend_sycl.h"
#include "utils_ranges_sycl.h"

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

template <typename... _Name>
class __reduce_small_kernel;

template <typename... _Name>
class __reduce_mid_device_kernel;

template <typename... _Name>
class __reduce_mid_work_group_kernel;

template <typename... _Name>
class __reduce_kernel;

// Single work group kernel that transforms and reduces __n elements to the single result.
template <typename _Tp, typename _NDItemId, typename _Size, typename _TransformPattern, typename _ReducePattern,
          typename _InitType, typename _AccLocal, typename _Res, typename... _Acc>
void
__work_group_reduce_kernel(const _NDItemId __item_id, const _Size __n, const _Size __n_items,
                           _TransformPattern __transform_pattern, _ReducePattern __reduce_pattern, _InitType __init,
                           const _AccLocal& __local_mem, const _Res& __res_acc, const _Acc&... __acc)
{
    auto __local_idx = __item_id.get_local_id(0);
    // 1. Initialization (transform part). Fill local memory
    __transform_pattern(__item_id, __n, /*global_offset*/ (_Size)0, __local_mem, __acc...);
    __dpl_sycl::__group_barrier(__item_id);
    // 2. Reduce within work group using local memory
    _Tp __result = __reduce_pattern(__item_id, __n_items, __local_mem);
    if (__local_idx == 0)
    {
        __reduce_pattern.apply_init(__init, __result);
        __res_acc[0] = __result;
    }
}

// Device kernel that transforms and reduces __n elements to the number of work groups preliminary results.
template <typename _Tp, typename _NDItemId, typename _Size, typename _TransformPattern, typename _ReducePattern,
          typename _AccLocal, typename _Tmp, typename... _Acc>
void
__device_reduce_kernel(const _NDItemId __item_id, const _Size __n, const _Size __n_items,
                       _TransformPattern __transform_pattern, _ReducePattern __reduce_pattern,
                       const _AccLocal& __local_mem, const _Tmp& __temp_acc, const _Acc&... __acc)
{
    auto __local_idx = __item_id.get_local_id(0);
    auto __group_idx = __item_id.get_group(0);
    // 1. Initialization (transform part). Fill local memory
    __transform_pattern(__item_id, __n, /*global_offset*/ (_Size)0, __local_mem, __acc...);
    __dpl_sycl::__group_barrier(__item_id);
    // 2. Reduce within work group using local memory
    _Tp __result = __reduce_pattern(__item_id, __n_items, __local_mem);
    if (__local_idx == 0)
        __temp_acc[__group_idx] = __result;
}

//------------------------------------------------------------------------
// parallel_transform_reduce - async patterns
// Please see the comment for __parallel_for_submitter for optional kernel name explanation
//------------------------------------------------------------------------

// Parallel_transform_reduce for a small arrays using a single sub group.
// Transforms and reduces __sub_group_size * __iters_per_work_item elements.
template <::std::uint16_t __sub_group_size, ::std::uint8_t __iters_per_work_item, typename _Tp>
struct __parallel_reduce_sub_group_submitter
{
    template <typename _ReduceOp, typename _TransformOp, typename _Size, typename _InitType>
    sycl::event
    operator()(sycl::queue Q, const _Size __n, _ReduceOp __reduce_op, _TransformOp __transform_op, _InitType __init,
               sycl::buffer<_Tp> __input, sycl::buffer<_Tp> __res) const
    {
        return Q.submit([&, __n](sycl::handler& __cgh) {
            auto __input_acc = __input.template get_access<sycl::access::mode::read>(__cgh);
            auto __res_acc = __res.template get_access<sycl::access::mode::write>(__cgh);
            __cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(__sub_group_size), sycl::range<1>(__sub_group_size)),
                [=](sycl::nd_item<1> __item_id) [[sycl::reqd_work_group_size(__sub_group_size), sycl::reqd_sub_group_size(__sub_group_size)]] {
                    unseq_backend::reduce_sub_group_kernel<__sub_group_size, __iters_per_work_item, _ReduceOp, _TransformOp, _Tp>(__item_id, __n, 0ul, __input_acc, __res_acc);
                });
        });
    }
}; // struct __parallel_transform_reduce_sub_group_submitter

// Parallel_transform_reduce for an array which fits in a single work group
// Transforms and reduces __sub_group_size * __iters_per_work_item elements.
template <::std::uint16_t __work_group_size, ::std::uint16_t __sub_group_size, ::std::uint8_t __iters_per_work_item, typename _Tp>
struct __parallel_reduce_single_work_group_submitter
{
    template <typename _ReduceOp, typename _TransformOp, typename _Size, typename _InitType>
    sycl::event
    operator()(sycl::queue Q, const _Size __n, _ReduceOp __reduce_op, _TransformOp __transform_op, _InitType __init,
               sycl::buffer<_Tp> __input, sycl::buffer<_Tp> __res) const
    {
        return Q.submit([&, __n](sycl::handler& __cgh) {
            auto __input_acc = __input.template get_access<sycl::access::mode::read>(__cgh);
            auto __res_acc = __res.template get_access<sycl::access::mode::write>(__cgh);
            sycl::local_accessor<int> local_acc(__work_group_size/__sub_group_size, __cgh);
            __cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(__work_group_size), sycl::range<1>(__work_group_size)),
                [=](sycl::nd_item<1> __item_id) [[sycl::reqd_work_group_size(__work_group_size), sycl::reqd_sub_group_size(__sub_group_size)]] {
                    unseq_backend::reduce_work_group_kernel<__sub_group_size, __iters_per_work_item, _ReduceOp, 
                                                                  _TransformOp, _Tp>(__item_id, __n, 0ul, local_acc, __input_acc, __res_acc);
                });
        });
    }
}; // struct __parallel_transform_reduce_single_work_group_submitter

// Parallel_transform_reduce for an array across multiple work gropus
// Transforms and reduces __sub_group_size * __iters_per_work_item elements per work group
// leading to ceil(__n/(__sub_group * __iters_per_work_item)) outputs
template <::std::uint16_t __work_group_size, ::std::uint16_t __sub_group_size, ::std::uint8_t __iters_per_work_item, typename _Tp>
struct __parallel_reduce_multi_work_group_submitter
{
    template <typename _ReduceOp, typename _TransformOp, typename _Size, typename _InitType>
    sycl::event
    operator()(sycl::queue Q, const _Size __n, _ReduceOp __reduce_op, _TransformOp __transform_op, _InitType __init,
               sycl::buffer<_Tp> __input, sycl::buffer<_Tp> __res) const
    {
        const _Size __global_size = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __work_group_size * __iters_per_work_item) * __work_group_size;
        return Q.submit([&, __n](sycl::handler& __cgh) {
            auto __input_acc = __input.template get_access<sycl::access::mode::read>(__cgh);
            auto __res_acc = __res.template get_access<sycl::access::mode::write>(__cgh);
            sycl::local_accessor<int> local_acc(__work_group_size/__sub_group_size, __cgh);
            __cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(__global_size), sycl::range<1>(__work_group_size)),
                [=](sycl::nd_item<1> __item_id) [[sycl::reqd_work_group_size(__work_group_size), sycl::reqd_sub_group_size(__sub_group_size)]] {
                    unseq_backend::reduce_work_group_kernel<__sub_group_size, __iters_per_work_item, _ReduceOp, 
                                                                  _TransformOp, _Tp>(__item_id, __n, 0ul, local_acc, __input_acc, __res_acc);
                });
        });
    }
}; // struct __parallel_transform_reduce_multi_work_group_submitter

template <::std::uint16_t __work_group_size, ::std::uint16_t __sub_group_size, typename _Tp,
           typename _ReduceOp, typename _TransformOp, typename _Size, typename _InitType>
sycl::event __reduce_iter(sycl::queue Q, const _Size __n, _ReduceOp __reduce_op, _TransformOp __transform_op, _InitType __init,
                          sycl::buffer<_Tp> __input, sycl::buffer<_Tp> __res) {
    constexpr ::std::uint8_t __sub_group_iters_per_item = 64;
    constexpr ::std::uint8_t __work_group_iters_per_item = 32;

    constexpr _Size sub_group_limit = __sub_group_size * __sub_group_iters_per_item;
    constexpr _Size work_group_limit = __work_group_size * __work_group_iters_per_item;
    static_assert(sub_group_limit < work_group_limit);

    sycl::event __e;
    if(__n < sub_group_limit) {
        // Sub group reduce
        __parallel_reduce_sub_group_submitter<__sub_group_size, __sub_group_iters_per_item, _Tp> __sub_group_reduce;
        __e = __sub_group_reduce(Q, __n, __reduce_op, __transform_op, __init, __input, __res);
    } else if (__n < work_group_limit) {
        // Single work item reduce
        __parallel_reduce_single_work_group_submitter<__work_group_size, __sub_group_size, __work_group_iters_per_item, _Tp> __work_group_reduce;
        __e = __work_group_reduce(Q, __n, __reduce_op, __transform_op, __init, __input, __res);
    } else {
        // Multi work item reduce
        __parallel_reduce_multi_work_group_submitter<__work_group_size, __sub_group_size, __work_group_iters_per_item, _Tp> __multi_group_reduce;
        __e = __multi_group_reduce(Q, __n, __reduce_op, __transform_op, __init, __input, __res);
    }
    return __e;
}


// Helper used for calculating scratchpad memory
template <::std::uint16_t __work_group_size, typename _Size>
_Size __calc_reduce_output_size(const _Size& __n) {
    constexpr ::std::uint8_t __work_group_iters_per_item = 32;
    _Size work_group_limit = __work_group_size * __work_group_iters_per_item;

    // TODO: see if __work_group_iters_per_item can be combined with __reduce_iter
    if(__n < work_group_limit) 
        // Sub group or work item reduce
        return 1;

    // Multi work item reduce
    return oneapi::dpl::__internal::__dpl_ceiling_div(__n, __work_group_size * __work_group_iters_per_item);
}

// Internally used to calculate number of __reduce_itter calls needed
template <::std::uint16_t __work_group_size, typename _Size>
_Size __calc_reduce_passes(const _Size& __n) {
    _Size __output_size = __n;

    _Size n_iter = 0;
    while(__output_size > 1){
        __output_size = __calc_reduce_output_size<__work_group_size>(__n);
        ++n_iter;
    }
    return n_iter;
}

// Internally used to allocate scratchpad memory
template <::std::uint16_t __work_group_size, typename _Size>
_Size __calc_reduce_scratch_size(const _Size& __n, const _Size& __n_passes) {
   if (__n_passes == 2) {
       return __calc_reduce_output_size<__work_group_size>(__n);
   } else if (__n_passes > 2) {
       _Size __intermediate_n = __calc_reduce_output_size<__work_group_size>(__n);
       return __calc_reduce_output_size<__work_group_size>(__intermediate_n) + __intermediate_n;
   }
   return 0; // 1 pass
}

// Handles calls to __reduce_iter and scratch memory
template <::std::uint16_t __work_group_size, ::std::uint16_t __sub_group_size, typename _Tp,
           typename _ReduceOp, typename _TransformOp, typename _Size, typename _InitType>
sycl::event __reduce_driver(sycl::queue Q, const _Size __n, _ReduceOp __reduce_op, _TransformOp __transform_op, _InitType __init,
                          sycl::buffer<_Tp> __input, sycl::buffer<_Tp> __res) {
    _Size __n_passes = __calc_reduce_passes<__work_group_size>(__n);

    // Get scratch memory splits
    // TODO: make USM and offset
    _Size __scratchpad_size_total = __calc_reduce_scratch_size<__work_group_size>(__n, __n_passes);
    _Size __scratchpad_size_1 = __calc_reduce_output_size<__work_group_size>(__n);

    sycl::buffer<_Tp> __scratch_mem_1((__n_passes > 1) ? __scratchpad_size_1 : 1);
    sycl::buffer<_Tp> __scratch_mem_2((__n_passes > 2) ? __calc_reduce_output_size<__work_group_size>(__n) : 1);

    sycl::event e;
    sycl::buffer<_Tp> __iter_in_ptr = __input;
    sycl::buffer<_Tp> __iter_out_ptr = __scratch_mem_1;
    _Size __iter_n = __n;
    for (int i = 0; i < __n_passes; ++i) {
        if (i > 0) {
            // if not first pass
            __iter_in_ptr = __iter_out_ptr;
            __iter_out_ptr = (__iter_in_ptr == __scratch_mem_1) ? __scratch_mem_2 : __scratch_mem_1;
        }
        if (i == __n_passes - 1) // if last iter, set final output
            __iter_out_ptr = __res;

        __reduce_iter<__work_group_size, __sub_group_size, _Tp>(Q, __iter_n, __reduce_op, __transform_op, __init, __input, __res);
        __iter_n = __calc_reduce_output_size<__work_group_size>(__iter_n);
    }
    return e;
}

// Parallel_transform_reduce for a small arrays using a single work group.
// Transforms and reduces __work_group_size * __iters_per_work_item elements.
template <::std::uint16_t __work_group_size, ::std::uint8_t __iters_per_work_item, typename _Tp, typename _KernelName>
struct __parallel_transform_reduce_small_submitter;

template <::std::uint16_t __work_group_size, ::std::uint8_t __iters_per_work_item, typename _Tp, typename... _Name>
struct __parallel_transform_reduce_small_submitter<__work_group_size, __iters_per_work_item, _Tp,
                                                   __internal::__optional_kernel_name<_Name...>>
{
    template <typename _ExecutionPolicy, typename _ReduceOp, typename _TransformOp, typename _Size, typename _InitType,
              oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0,
              typename... _Ranges>
    auto
    operator()(_ExecutionPolicy&& __exec, const _Size __n, _ReduceOp __reduce_op, _TransformOp __transform_op,
               _InitType __init, _Ranges&&... __rngs) const
    {
        auto __transform_pattern =
            unseq_backend::transform_reduce<_ExecutionPolicy, __iters_per_work_item, _ReduceOp, _TransformOp>{
                __reduce_op, __transform_op};
        auto __reduce_pattern = unseq_backend::reduce_over_group<_ExecutionPolicy, _ReduceOp, _Tp>{__reduce_op};

        const _Size __n_items = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __iters_per_work_item);

        sycl::buffer<_Tp> __res(sycl::range<1>(1));

        sycl::event __reduce_event = __exec.queue().submit([&, __n, __n_items](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rngs...); // get an access to data under SYCL buffer
            sycl::accessor __res_acc{__res, __cgh, sycl::write_only, __dpl_sycl::__no_init{}};
            __dpl_sycl::__local_accessor<_Tp> __temp_local(sycl::range<1>(__work_group_size), __cgh);
            __cgh.parallel_for<_Name...>(
                sycl::nd_range<1>(sycl::range<1>(__work_group_size), sycl::range<1>(__work_group_size)),
                [=](sycl::nd_item<1> __item_id) {
                    __work_group_reduce_kernel<_Tp>(__item_id, __n, __n_items, __transform_pattern, __reduce_pattern,
                                                    __init, __temp_local, __res_acc, __rngs...);
                });
        });
        return __future(__reduce_event, __res);
    }
}; // struct __parallel_transform_reduce_small_submitter

template <::std::uint16_t __work_group_size, ::std::uint8_t __iters_per_work_item, typename _Tp, typename _ReduceOp,
          typename _TransformOp, typename _ExecutionPolicy, typename _Size, typename _InitType,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0, typename... _Ranges>
auto
__parallel_transform_reduce_small_impl(_ExecutionPolicy&& __exec, const _Size __n, _ReduceOp __reduce_op,
                                       _TransformOp __transform_op, _InitType __init, _Ranges&&... __rngs)
{
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using _CustomName = typename _Policy::kernel_name;
    using _ReduceKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __reduce_small_kernel<::std::integral_constant<::std::uint8_t, __iters_per_work_item>, _CustomName>>;

    return __parallel_transform_reduce_small_submitter<__work_group_size, __iters_per_work_item, _Tp, _ReduceKernel>()(
        ::std::forward<_ExecutionPolicy>(__exec), __n, __reduce_op, __transform_op, __init,
        ::std::forward<_Ranges>(__rngs)...);
}

// Submits the first kernel of the parallel_transform_reduce for mid-sized arrays.
// Uses multiple work groups that each reduce __work_group_size * __iters_per_work_item items and store the preliminary
// results in __temp.
template <::std::uint16_t __work_group_size, ::std::uint8_t __iters_per_work_item, typename _Tp, typename _KernelName>
struct __parallel_transform_reduce_device_kernel_submitter;

template <::std::uint16_t __work_group_size, ::std::uint8_t __iters_per_work_item, typename _Tp,
          typename... _KernelName>
struct __parallel_transform_reduce_device_kernel_submitter<__work_group_size, __iters_per_work_item, _Tp,
                                                           __internal::__optional_kernel_name<_KernelName...>>
{
    template <typename _ExecutionPolicy, typename _ReduceOp, typename _TransformOp, typename _Size, typename _InitType,
              oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0,
              typename... _Ranges>
    auto
    operator()(_ExecutionPolicy&& __exec, _Size __n, _ReduceOp __reduce_op, _TransformOp __transform_op,
               _InitType __init, sycl::buffer<_Tp>& __temp, _Ranges&&... __rngs) const
    {
        auto __transform_pattern =
            unseq_backend::transform_reduce<_ExecutionPolicy, __iters_per_work_item, _ReduceOp, _TransformOp>{
                __reduce_op, __transform_op};
        auto __reduce_pattern = unseq_backend::reduce_over_group<_ExecutionPolicy, _ReduceOp, _Tp>{__reduce_op};

        // number of buffer elements processed within workgroup
        constexpr _Size __size_per_work_group = __iters_per_work_item * __work_group_size;
        const _Size __n_groups = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __size_per_work_group);
        _Size __n_items = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __iters_per_work_item);

        return __exec.queue().submit([&, __n, __n_items](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rngs...); // get an access to data under SYCL buffer
            sycl::accessor __temp_acc{__temp, __cgh, sycl::write_only, __dpl_sycl::__no_init{}};
            __dpl_sycl::__local_accessor<_Tp> __temp_local(sycl::range<1>(__work_group_size), __cgh);
            __cgh.parallel_for<_KernelName...>(
                sycl::nd_range<1>(sycl::range<1>(__n_groups * __work_group_size), sycl::range<1>(__work_group_size)),
                [=](sycl::nd_item<1> __item_id) {
                    __device_reduce_kernel<_Tp>(__item_id, __n, __n_items, __transform_pattern, __reduce_pattern,
                                                __temp_local, __temp_acc, __rngs...);
                });
        });
    }
}; // struct __parallel_transform_reduce_device_kernel_submitter

// Submits the second kernel of the parallel_transform_reduce for mid-sized arrays.
// Uses a single work groups to reduce __n preliminary results stored in __temp and returns a future object with the
// result buffer.
template <::std::uint16_t __work_group_size, ::std::uint8_t __iters_per_work_item, typename _Tp, typename _KernelName>
struct __parallel_transform_reduce_work_group_kernel_submitter;

template <::std::uint16_t __work_group_size, ::std::uint8_t __iters_per_work_item, typename _Tp,
          typename... _KernelName>
struct __parallel_transform_reduce_work_group_kernel_submitter<__work_group_size, __iters_per_work_item, _Tp,
                                                               __internal::__optional_kernel_name<_KernelName...>>
{
    template <typename _ExecutionPolicy, typename _ReduceOp, typename _TransformOp, typename _Size, typename _InitType,
              oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0>
    auto
    operator()(_ExecutionPolicy&& __exec, sycl::event& __reduce_event, _Size __n, _ReduceOp __reduce_op,
               _TransformOp __transform_op, _InitType __init, sycl::buffer<_Tp>& __temp) const
    {
        using _NoOpFunctor = unseq_backend::walk_n<_ExecutionPolicy, oneapi::dpl::__internal::__no_op>;
        auto __transform_pattern =
            unseq_backend::transform_reduce<_ExecutionPolicy, __iters_per_work_item, _ReduceOp, _NoOpFunctor>{
                __reduce_op, _NoOpFunctor{}};
        auto __reduce_pattern = unseq_backend::reduce_over_group<_ExecutionPolicy, _ReduceOp, _Tp>{__reduce_op};

        // Lower the work group size of the second kernel to the next power of 2 if __n < __work_group_size.
        auto __work_group_size2 = __work_group_size;
        if constexpr (__iters_per_work_item == 1)
        {
            if (__n < __work_group_size)
            {
                __work_group_size2 = __n;
                if ((__work_group_size2 & (__work_group_size2 - 1)) != 0)
                    __work_group_size2 = oneapi::dpl::__internal::__dpl_bit_floor(__work_group_size2) << 1;
            }
        }
        const _Size __n_items = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __iters_per_work_item);

        sycl::buffer<_Tp> __res(sycl::range<1>(1));

        __reduce_event = __exec.queue().submit([&, __n, __n_items](sycl::handler& __cgh) {
            __cgh.depends_on(__reduce_event);

            sycl::accessor __temp_acc{__temp, __cgh, sycl::read_only};
            sycl::accessor __res_acc{__res, __cgh, sycl::write_only, __dpl_sycl::__no_init{}};
            __dpl_sycl::__local_accessor<_Tp> __temp_local(sycl::range<1>(__work_group_size2), __cgh);

            __cgh.parallel_for<_KernelName...>(
                sycl::nd_range<1>(sycl::range<1>(__work_group_size2), sycl::range<1>(__work_group_size2)),
                [=](sycl::nd_item<1> __item_id) {
                    __work_group_reduce_kernel<_Tp>(__item_id, __n, __n_items, __transform_pattern, __reduce_pattern,
                                                    __init, __temp_local, __res_acc, __temp_acc);
                });
        });

        return __future(__reduce_event, __res);
    }
}; // struct __parallel_transform_reduce_work_group_kernel_submitter

template <::std::uint16_t __work_group_size, ::std::uint8_t __iters_per_work_item_device_kernel,
          ::std::uint8_t __iters_per_work_item_work_group_kernel, typename _Tp, typename _ReduceOp,
          typename _TransformOp, typename _ExecutionPolicy, typename _Size, typename _InitType,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0, typename... _Ranges>
auto
__parallel_transform_reduce_mid_impl(_ExecutionPolicy&& __exec, _Size __n, _ReduceOp __reduce_op,
                                     _TransformOp __transform_op, _InitType __init, _Ranges&&... __rngs)
{
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using _CustomName = typename _Policy::kernel_name;

    // The same value for __iters_per_work_item_device_kernel is currently used. Include
    // __iters_per_work_item_device_kernel in case this changes in the future.
    using _ReduceDeviceKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__reduce_mid_device_kernel<_CustomName>>;
    using _ReduceWorkGroupKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__reduce_mid_work_group_kernel<
            ::std::integral_constant<::std::uint8_t, __iters_per_work_item_work_group_kernel>, _CustomName>>;

    // number of buffer elements processed within workgroup
    constexpr _Size __size_per_work_group = __iters_per_work_item_device_kernel * __work_group_size;
    const _Size __n_groups = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __size_per_work_group);
    sycl::buffer<_Tp> __temp{sycl::range<1>(__n_groups)};

    sycl::event __reduce_event =
        __parallel_transform_reduce_device_kernel_submitter<__work_group_size, __iters_per_work_item_device_kernel, _Tp,
                                                            _ReduceDeviceKernel>()(
            __exec, __n, __reduce_op, __transform_op, __init, __temp, ::std::forward<_Ranges>(__rngs)...);

    __n = __n_groups; // Number of preliminary results from the device kernel.
    return __parallel_transform_reduce_work_group_kernel_submitter<
        __work_group_size, __iters_per_work_item_work_group_kernel, _Tp, _ReduceWorkGroupKernel>()(
        ::std::forward<_ExecutionPolicy>(__exec), __reduce_event, __n, __reduce_op, __transform_op, __init, __temp);
}

// General implementation using a tree reduction
template <typename _Tp, ::std::uint8_t __iters_per_work_item>
struct __parallel_transform_reduce_impl
{
    template <typename _ExecutionPolicy, typename _Size, typename _ReduceOp, typename _TransformOp, typename _InitType,
              oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0,
              typename... _Ranges>
    static auto
    submit(_ExecutionPolicy&& __exec, _Size __n, ::std::uint16_t __work_group_size, _ReduceOp __reduce_op,
           _TransformOp __transform_op, _InitType __init, _Ranges&&... __rngs)
    {
        using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
        using _CustomName = typename _Policy::kernel_name;
        using _NoOpFunctor = unseq_backend::walk_n<_ExecutionPolicy, oneapi::dpl::__internal::__no_op>;
        using _ReduceKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_generator<
            __reduce_kernel, _CustomName, _ReduceOp, _TransformOp, _NoOpFunctor, _Ranges...>;

        auto __transform_pattern1 =
            unseq_backend::transform_reduce<_ExecutionPolicy, __iters_per_work_item, _ReduceOp, _TransformOp>{
                __reduce_op, __transform_op};
        auto __transform_pattern2 =
            unseq_backend::transform_reduce<_ExecutionPolicy, __iters_per_work_item, _ReduceOp, _NoOpFunctor>{
                __reduce_op, _NoOpFunctor{}};
        auto __reduce_pattern = unseq_backend::reduce_over_group<_ExecutionPolicy, _ReduceOp, _Tp>{__reduce_op};

#if _ONEDPL_COMPILE_KERNEL
        auto __kernel = __internal::__kernel_compiler<_ReduceKernel>::__compile(__exec);
        __work_group_size = ::std::min(
            __work_group_size, (::std::uint16_t)oneapi::dpl::__internal::__kernel_work_group_size(__exec, __kernel));
#endif

        _Size __size_per_work_group =
            __iters_per_work_item * __work_group_size; // number of buffer elements processed within workgroup
        _Size __n_groups = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __size_per_work_group);
        _Size __n_items = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __iters_per_work_item);

        // Create temporary global buffers to store temporary values
        sycl::buffer<_Tp> __temp(sycl::range<1>(2 * __n_groups));
        sycl::buffer<_Tp> __res(sycl::range<1>(1));
        // __is_first == true. Reduce over each work_group
        // __is_first == false. Reduce between work groups
        bool __is_first = true;

        // For memory utilization it's better to use one big buffer instead of two small because size of the buffer is
        // close to a few MB
        _Size __offset_1 = 0;
        _Size __offset_2 = __n_groups;

        sycl::event __reduce_event;
        do
        {
            __reduce_event = __exec.queue().submit([&, __is_first, __offset_1, __offset_2, __n, __n_items,
                                                    __n_groups](sycl::handler& __cgh) {
                __cgh.depends_on(__reduce_event);

                oneapi::dpl::__ranges::__require_access(__cgh, __rngs...); // get an access to data under SYCL buffer
                sycl::accessor __temp_acc{__temp, __cgh, sycl::read_write};
                sycl::accessor __res_acc{__res, __cgh, sycl::write_only, __dpl_sycl::__no_init{}};
                __dpl_sycl::__local_accessor<_Tp> __temp_local(sycl::range<1>(__work_group_size), __cgh);
#if _ONEDPL_COMPILE_KERNEL && _ONEDPL_KERNEL_BUNDLE_PRESENT
                __cgh.use_kernel_bundle(__kernel.get_kernel_bundle());
#endif
                __cgh.parallel_for<_ReduceKernel>(
#if _ONEDPL_COMPILE_KERNEL && !_ONEDPL_KERNEL_BUNDLE_PRESENT
                    __kernel,
#endif
                    sycl::nd_range<1>(sycl::range<1>(__n_groups * __work_group_size),
                                      sycl::range<1>(__work_group_size)),
                    [=](sycl::nd_item<1> __item_id) {
                        auto __local_idx = __item_id.get_local_id(0);
                        auto __group_idx = __item_id.get_group(0);
                        // 1. Initialization (transform part). Fill local memory
                        if (__is_first)
                            __transform_pattern1(__item_id, __n, /*global_offset*/ (_Size)0, __temp_local, __rngs...);
                        else
                            __transform_pattern2(__item_id, __n, __offset_2, __temp_local, __temp_acc);
                        __dpl_sycl::__group_barrier(__item_id);
                        // 2. Reduce within work group using local memory
                        _Tp __result = __reduce_pattern(__item_id, __n_items, __temp_local);
                        if (__local_idx == 0)
                        {
                            // final reduction
                            if (__n_groups == 1)
                            {
                                __reduce_pattern.apply_init(__init, __result);
                                __res_acc[0] = __result;
                            }

                            __temp_acc[__offset_1 + __group_idx] = __result;
                        }
                    });
            });
            if (__is_first)
                __is_first = false;
            ::std::swap(__offset_1, __offset_2);
            __n = __n_groups;
            __n_items = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __iters_per_work_item);
            __n_groups = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __size_per_work_group);
        } while (__n > 1);

        return __future(__reduce_event, __res);
    }
}; // struct __parallel_transform_reduce_impl

// General version of parallel_transform_reduce.
// The binary operator must be associative but commutativity is not required since the elements are processed in order.
// Each work item transforms and reduces __iters_per_work_item elements from global memory and stores the result in SLM.
// 32 __iters_per_work_item was empirically found best for typical devices.
// Each work group of size __work_group_size reduces the preliminary results of each work item in a group reduction
// using SLM. 256 __work_group_size was empirically found best for typical devices.
// A single-work group implementation is used for small arrays.
// Mid-sized arrays use two tree reductions with independent __iters_per_work_item.
// Big arrays are processed with a recursive tree reduction. __work_group_size * __iters_per_work_item elements are
// reduced in each step.
template <typename _Tp, typename _ReduceOp, typename _TransformOp, typename _ExecutionPolicy, typename _InitType,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0, typename... _Ranges>
auto
__parallel_transform_reduce_1(_ExecutionPolicy&& __exec, _ReduceOp __reduce_op, _TransformOp __transform_op,
                            _InitType __init, std::false_type /*has_known_identity*/, _Ranges&&... __rngs)
{
    auto __n = oneapi::dpl::__ranges::__get_first_range_size(__rngs...);
    assert(__n > 0);

    // Get the work group size adjusted to the local memory limit.
    // Pessimistically double the memory requirement to take into account memory used by compiled kernel.
    // TODO: find a way to generalize getting of reliable work-group size.
    ::std::size_t __work_group_size = oneapi::dpl::__internal::__slm_adjusted_work_group_size(__exec, sizeof(_Tp) * 2);

    // Use single work group implementation if array < __work_group_size * __iters_per_work_item.
    if (__work_group_size >= 256)
    {
        if (__n <= 256)
        {
            return __parallel_transform_reduce_small_impl<256, 1, _Tp>(::std::forward<_ExecutionPolicy>(__exec), __n,
                                                                       __reduce_op, __transform_op, __init,
                                                                       ::std::forward<_Ranges>(__rngs)...);
        }
        else if (__n <= 512)
        {
            return __parallel_transform_reduce_small_impl<256, 2, _Tp>(::std::forward<_ExecutionPolicy>(__exec), __n,
                                                                       __reduce_op, __transform_op, __init,
                                                                       ::std::forward<_Ranges>(__rngs)...);
        }
        else if (__n <= 1024)
        {
            return __parallel_transform_reduce_small_impl<256, 4, _Tp>(::std::forward<_ExecutionPolicy>(__exec), __n,
                                                                       __reduce_op, __transform_op, __init,
                                                                       ::std::forward<_Ranges>(__rngs)...);
        }
        else if (__n <= 2048)
        {
            return __parallel_transform_reduce_small_impl<256, 8, _Tp>(::std::forward<_ExecutionPolicy>(__exec), __n,
                                                                       __reduce_op, __transform_op, __init,
                                                                       ::std::forward<_Ranges>(__rngs)...);
        }
        else if (__n <= 4096)
        {
            return __parallel_transform_reduce_small_impl<256, 16, _Tp>(::std::forward<_ExecutionPolicy>(__exec), __n,
                                                                        __reduce_op, __transform_op, __init,
                                                                        ::std::forward<_Ranges>(__rngs)...);
        }
        else if (__n <= 8192)
        {
            return __parallel_transform_reduce_small_impl<256, 32, _Tp>(::std::forward<_ExecutionPolicy>(__exec), __n,
                                                                        __reduce_op, __transform_op, __init,
                                                                        ::std::forward<_Ranges>(__rngs)...);
        }

        // Use two-step tree reduction.
        // First step reduces __work_group_size * __iters_per_work_item_device_kernel elements.
        // Second step reduces __work_group_size * __iters_per_work_item_work_group_kernel elements.
        else if (__n <= 2097152)
        {
            return __parallel_transform_reduce_mid_impl<256, 32, 1, _Tp>(::std::forward<_ExecutionPolicy>(__exec), __n,
                                                                         __reduce_op, __transform_op, __init,
                                                                         ::std::forward<_Ranges>(__rngs)...);
        }
        else if (__n <= 4194304)
        {
            return __parallel_transform_reduce_mid_impl<256, 32, 2, _Tp>(::std::forward<_ExecutionPolicy>(__exec), __n,
                                                                         __reduce_op, __transform_op, __init,
                                                                         ::std::forward<_Ranges>(__rngs)...);
        }
        else if (__n <= 8388608)
        {
            return __parallel_transform_reduce_mid_impl<256, 32, 4, _Tp>(::std::forward<_ExecutionPolicy>(__exec), __n,
                                                                         __reduce_op, __transform_op, __init,
                                                                         ::std::forward<_Ranges>(__rngs)...);
        }
        else if (__n <= 16777216)
        {
            return __parallel_transform_reduce_mid_impl<256, 32, 8, _Tp>(::std::forward<_ExecutionPolicy>(__exec), __n,
                                                                         __reduce_op, __transform_op, __init,
                                                                         ::std::forward<_Ranges>(__rngs)...);
        }
        else if (__n <= 33554432)
        {
            return __parallel_transform_reduce_mid_impl<256, 32, 16, _Tp>(::std::forward<_ExecutionPolicy>(__exec), __n,
                                                                          __reduce_op, __transform_op, __init,
                                                                          ::std::forward<_Ranges>(__rngs)...);
        }
        else if (__n <= 67108864)
        {
            return __parallel_transform_reduce_mid_impl<256, 32, 32, _Tp>(::std::forward<_ExecutionPolicy>(__exec), __n,
                                                                          __reduce_op, __transform_op, __init,
                                                                          ::std::forward<_Ranges>(__rngs)...);
        }
    }
    // Otherwise use a recursive tree reduction.
    return __parallel_transform_reduce_impl<_Tp, 32>::submit(::std::forward<_ExecutionPolicy>(__exec), __n,
                                                             __work_group_size, __reduce_op, __transform_op, __init,
                                                             ::std::forward<_Ranges>(__rngs)...);
}

template <typename _Tp, typename _ReduceOp, typename _TransformOp, typename _ExecutionPolicy, typename _InitType,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0, typename... _Ranges>
auto
__parallel_transform_reduce_1(_ExecutionPolicy&& __exec, _ReduceOp __reduce_op, _TransformOp __transform_op,
                            _InitType __init, std::true_type /*has_known_identity*/, _Ranges&&... __rngs)
{
    auto __n = oneapi::dpl::__ranges::__get_first_range_size(__rngs...);
    __reduce_driver<256, 32, _Tp>(__exec.queue(), __n, __reduce_op, __transform_op, __init, __rngs...);
}

template <typename _Tp, typename _ReduceOp, typename _TransformOp, typename _ExecutionPolicy, typename _InitType,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0, typename... _Ranges>
auto
__parallel_transform_reduce(_ExecutionPolicy&& __exec, _ReduceOp __reduce_op, _TransformOp __transform_op,
                            _InitType __init, _Ranges&&... __rngs) {
    return __parallel_transform_reduce_1<_Tp>(__exec, __reduce_op, __transform_op, __init, unseq_backend::__has_known_identity<_ReduceOp, _Tp>{}, __rngs...);
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_H
