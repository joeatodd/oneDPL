// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#ifndef _ONEDPL_parallel_backend_sycl_scan_H
#define _ONEDPL_parallel_backend_sycl_scan_H

namespace oneapi::dpl::experimental::kt
{

inline namespace igpu {

template<typename _T>
struct __scan_status_flag
{
    // xxxx00 - not computed
    // xxxx01 - partial
    // xxxx10 - full
    // xxx100 - out of bounds

    using _AtomicRefT = sycl::atomic_ref<::std::uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>;
    static constexpr std::uint32_t partial_mask = 1;
    static constexpr std::uint32_t full_mask = 2;
    static constexpr std::uint32_t oob_value = 4;

    static constexpr int padding = 32;

    __scan_status_flag(std::uint32_t* flags_begin, const std::uint32_t tile_id, _T* sums)
        : atomic_flag(*(flags_begin + tile_id + padding)), scanned_value(sums + tile_id + padding)
    {
    }

    // Change
    void
    set_partial(_T val)
    {
        (*scanned_value) = val;
        atomic_flag.store(partial_mask);
    }

    void
    set_full(_T val)
    {
        // :garbage:
        (*scanned_value) = val;
        atomic_flag.store(full_mask);
    }

    template <typename _Subgroup, typename BinOp>
    _T
    cooperative_lookback(std::uint32_t tile_id, const _Subgroup& subgroup, BinOp bin_op, std::uint32_t* flags_begin,
                         _T* sums)
    {
        _T sum = 0;
        int offset = -1;
        int i = 0;
        int local_id = subgroup.get_local_id();

        for (int tile = static_cast<int>(tile_id) + offset; tile >= 0; tile -= 32)
        {
            _AtomicRefT tile_atomic(*(flags_begin + tile + padding - local_id));

            std::uint32_t flag = 0;
            do
            {
                flag = tile_atomic.load();
            } while (!sycl::all_of_group(subgroup, flag != 0));

            bool is_full = flag & full_mask;
            auto is_full_ballot = sycl::ext::oneapi::group_ballot(subgroup, is_full);
            ::std::uint32_t is_full_ballot_bits{};
            is_full_ballot.extract_bits(is_full_ballot_bits);

            auto lowest_item_with_full = sycl::ctz(is_full_ballot_bits);

            _T* tile_val{sums + tile + padding - local_id};
            _T contribution = local_id <= lowest_item_with_full ? *tile_val : _T{0};

            // Sum all of the partial results from the tiles found, as well as the full contribution from the closest tile (if any)
            sum += sycl::reduce_over_group(subgroup, contribution, bin_op);

            // If we found a full value, we can stop looking at previous tiles. Otherwise,
            // keep going through tiles until we either find a full tile or we've completely
            // recomputed the prefix using partial values
            if (is_full_ballot_bits)
                break;

            //if (i++ > 10) break;
        }
        return sum;
    }

#if 0
    _T lookback(const std::uint32_t tile_id, std::uint32_t* flags_begin)
    {
        _T sum = 0;
        int i = 0;
        for (std::int32_t tile = static_cast<std::int32_t>(tile_id) - 1; tile >= 0; --tile)
        {
            _AtomicRefT tile_atomic(*(flags_begin + tile + padding));
            std::uint32_t tile_val = 0;
            do {
                tile_val = tile_atomic.load();
            } while (tile_val == 0);

            sum += tile_val & value_mask;

            // If this was a full value, we can stop looking at previous tiles. Otherwise,
            // keep going through tiles until we either find a full tile or we've completely
            // recomputed the prefix using partial values
            if (tile_val & full_mask)
                break;
        }
        return sum;
    }
#endif

    _AtomicRefT atomic_flag;
    _T* scanned_value;
};

template <typename _KernelParam, bool _Inclusive, typename _InRange, typename _OutRange, typename _BinaryOp>
void
single_pass_scan_impl(sycl::queue __queue, _InRange&& __in_rng, _OutRange&& __out_rng, _BinaryOp __binary_op)
{
    using _Type = oneapi::dpl::__internal::__value_t<_InRange>;

    static_assert(_Inclusive, "Single-pass scan only available for inclusive scan");

    const ::std::size_t n = __in_rng.size();
    // auto __max_cu = __queue.get_device().template get_info<sycl::info::device::max_compute_units>();
    //std::size_t num_wgs = __max_cu;
    //std::size_t num_wgs = 448;
    //std::size_t num_wgs = 256;

    // TODO: use wgsize and iters per item from _KernelParam
    //constexpr ::std::size_t __elems_per_workload = _KernelParam::data_per_workitem;
#ifdef _ONEDPL_SCAN_ITER_SIZE
    constexpr ::std::size_t __elems_per_workload = _ONEDPL_SCAN_ITER_SIZE;
#else
    constexpr ::std::size_t __elems_per_workload = 8;
#endif
    // Next power of 2 greater than or equal to __n
    auto __n_uniform = n;
    if ((__n_uniform & (__n_uniform - 1)) != 0)
        __n_uniform = oneapi::dpl::__internal::__dpl_bit_floor(n) << 1;
    //std::size_t wgsize = n/num_wgs/__elems_per_workload;
    std::size_t num_workloads = __n_uniform / __elems_per_workload;
    std::size_t wgsize = num_workloads > 256 ? 256 : num_workloads;
    std::size_t num_wgs = oneapi::dpl::__internal::__dpl_ceiling_div(num_workloads, wgsize);


    constexpr int status_flag_padding = 32;
    std::uint32_t status_flags_size = num_wgs+1+status_flag_padding;

    // printf("launching kernel num_workloads=%lu wgs=%lu wgsize=%lu elems_per_iter=%lu max_cu=%u\n", num_workloads,
    //  num_wgs, wgsize, __elems_per_workload, __max_cu);

    // One byte flags?
    uint32_t* status_flags = sycl::malloc_device<uint32_t>(status_flags_size, __queue);
    _Type* sums = sycl::malloc_device<_Type>(status_flags_size, __queue);
    auto fill_event_2 = __queue.fill<_Type>(sums, _Type{0}, status_flags_size);

    auto fill_event = __queue.submit([&](sycl::handler& hdl) {
        hdl.parallel_for<class scan_kt_init>(sycl::range<1>{status_flags_size}, [=](const sycl::item<1>& item)  {
                int id = item.get_linear_id();
                status_flags[id] = id < status_flag_padding ? __scan_status_flag<_Type>::oob_value : 0;
        });
    });


    std::uint32_t elems_in_tile = wgsize*__elems_per_workload;

#define SCAN_KT_DEBUG 0
#if SCAN_KT_DEBUG
    std::vector<uint32_t> debug11v(status_flags_size);
    __queue.memcpy(debug11v.data(), status_flags, status_flags_size * sizeof(uint32_t));

    for (int i = 0; i < status_flags_size-1; ++i)
        std::cout << "flag_before " << i << " " status_flags << debug11v[i] << std::endl;

    _Type* debug1 = sycl::malloc_device<_Type>(status_flags_size, __queue);
    uint32_t* debug2 = sycl::malloc_device<uint32_t>(status_flags_size, __queue);
    uint32_t* debug3 = sycl::malloc_device<uint32_t>(status_flags_size, __queue);
    uint32_t* debug4 = sycl::malloc_device<uint32_t>(status_flags_size, __queue);
    uint32_t* debug5 = sycl::malloc_device<uint32_t>(status_flags_size, __queue);
#endif

    auto event = __queue.submit([&](sycl::handler& hdl) {
        auto tile_id_lacc = sycl::local_accessor<std::uint32_t, 1>(sycl::range<1>{1}, hdl);
        hdl.depends_on(std::vector<sycl::event>{fill_event, fill_event_2});

        oneapi::dpl::__ranges::__require_access(hdl, __in_rng, __out_rng);
        hdl.parallel_for<class scan_kt_main>(sycl::nd_range<1>(num_workloads, wgsize), [=](const sycl::nd_item<1>& item)  [[intel::reqd_sub_group_size(32)]] {
            auto group = item.get_group();
            auto subgroup = item.get_sub_group();


            // Obtain unique ID for this work-group that will be used in decoupled lookback
            if (group.leader())
            {
                sycl::atomic_ref<::std::uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> idx_atomic(status_flags[status_flags_size-1]);
                tile_id_lacc[0] = idx_atomic.fetch_add(1);
            }
            sycl::group_barrier(group);
            std::uint32_t tile_id = tile_id_lacc[0];
#if SCAN_KT_DEBUG
            debug5[group.get_group_linear_id()] = tile_id;
#endif

            auto current_offset = (tile_id*elems_in_tile);
            auto next_offset = ((tile_id+1)*elems_in_tile);
            if (next_offset > n)
                next_offset = n;
            auto in_begin = __in_rng.begin() + current_offset;
            auto in_end = __in_rng.begin() + next_offset;
            auto out_begin = __out_rng.begin() + current_offset;


#if SCAN_KT_DEBUG
            debug3[tile_id] = current_offset;
            debug4[tile_id] = next_offset;
#endif

            if (current_offset >= n)
                return;

            auto local_sum = sycl::joint_reduce(group, in_begin, in_end, __binary_op);
#if SCAN_KT_DEBUG
            debug1[tile_id] = local_sum;
#endif

            _Type prev_sum = 0;

            // The first sub-group will query the previous tiles to find a prefix
            if (subgroup.get_group_id() == 0)
            {
                __scan_status_flag<_Type> flag(status_flags, tile_id, sums);

                // Modify this to separate value (local_sum) from flag.
                if (group.leader())
                    flag.set_partial(local_sum);

                // Find lowest work-item that has a full result (if any) and sum up subsequent partial results to obtain this tile's exclusive sum
                //sycl::reduce_over_group(item.get_subgroup())
                prev_sum = flag.cooperative_lookback(tile_id, subgroup, __binary_op, status_flags, sums);

                if (group.leader())
                    flag.set_full(prev_sum + local_sum);
            }

            prev_sum = sycl::group_broadcast(group, prev_sum, 0);
            sycl::joint_inclusive_scan(group, in_begin, in_end, out_begin, __binary_op, prev_sum);
        });
    });

    event.wait();

#if SCAN_KT_DEBUG
    std::vector<_Type> debug1v(status_flags_size);
    std::vector<uint32_t> debug2v(status_flags_size);
    std::vector<uint32_t> debug3v(status_flags_size);
    std::vector<uint32_t> debug4v(status_flags_size);
    std::vector<uint32_t> debug5v(status_flags_size);
    std::vector<uint32_t> debug6v(status_flags_size);
    std::vector<_Type> debug7v(status_flags_size);
    __queue.memcpy(debug1v.data(), debug1, status_flags_size * sizeof(_Type));
    __queue.memcpy(debug2v.data(), debug2, status_flags_size * sizeof(uint32_t));
    __queue.memcpy(debug3v.data(), debug3, status_flags_size * sizeof(uint32_t));
    __queue.memcpy(debug4v.data(), debug4, status_flags_size * sizeof(uint32_t));
    __queue.memcpy(debug5v.data(), debug5, status_flags_size * sizeof(uint32_t));
    __queue.memcpy(debug6v.data(), status_flags, status_flags_size * sizeof(uint32_t));
    __queue.memcpy(debug7v.data(), sums, status_flags_size * sizeof(_Type));
    __queue.wait();

    for (int i = 0; i < status_flags_size-1; ++i)
        std::cout << "tile " << i << " " << debug5v[i] << std::endl;
    for (int i = 0; i < status_flags_size-1; ++i)
        std::cout << "local_sum " << i << " " << debug1v[i] << std::endl;
    for (int i = 0; i < status_flags_size-1; ++i)
    {
        auto val = debug7v[i];
        int a = val / elems_in_tile;
        int b = val % elems_in_tile;
        std::cout << "flags " << i << " " << std::bitset<32>(debug6v[i]) << " (" << val << " = " << a << "/"
                  << elems_in_tile << "+" << b << ")" << std::endl;
    }
    for (int i = 0; i < status_flags_size-1; ++i)
        std::cout << "lookback " << i << " " << debug2v[i] << std::endl;
    for (int i = 0; i < status_flags_size-1; ++i)
        std::cout << "offset " << i << " " << debug3v[i] << std::endl;
    for (int i = 0; i < status_flags_size-1; ++i)
        std::cout << "end " << i << " " << debug4v[i] << std::endl;
    sycl::free(debug1, __queue);
    sycl::free(debug2, __queue);
    sycl::free(debug3, __queue);
    sycl::free(debug4, __queue);
    sycl::free(debug5, __queue);
#endif

    sycl::free(status_flags, __queue);
    sycl::free(sums, __queue);
}

// The generic structure for configuring a kernel
template <std::uint16_t DataPerWorkItem, std::uint16_t WorkGroupSize, typename KernelName>
struct kernel_param
{
    static constexpr std::uint16_t data_per_workitem = DataPerWorkItem;
    static constexpr std::uint16_t workgroup_size = WorkGroupSize;
    using kernel_name = KernelName;
};

template <typename _KernelParam, typename _InIterator, typename _OutIterator, typename _BinaryOp>
void
single_pass_inclusive_scan(sycl::queue __queue, _InIterator __in_begin, _InIterator __in_end, _OutIterator __out_begin, _BinaryOp __binary_op)
{
    auto __n = __in_end - __in_begin;

#if SCAN_KT_DEBUG
    using _Type = std::remove_pointer_t<_InIterator>;
    std::vector<_Type> in_debug(__n);
    __queue.memcpy(in_debug.data(), __in_begin, __n * sizeof(_Type));

    for (int i = 0; i < __n; ++i)
        std::cout << "input_before " << i << " " << in_debug[i] << std::endl;
#endif

    //printf("KERNEL_TEMPLATE %lu\n", __n);
    auto __keep1 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _InIterator>();
    auto __buf1 = __keep1(__in_begin, __in_end);
    auto __keep2 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _OutIterator>();
    auto __buf2 = __keep2(__out_begin, __out_begin + __n);

    single_pass_scan_impl<_KernelParam, true>(__queue, __buf1.all_view(), __buf2.all_view(), __binary_op);

#if SCAN_KT_DEBUG
    std::vector<_Type> in_debug2(__n);
    __queue.memcpy(in_debug2.data(), __in_begin, __n * sizeof(_Type));

    for (int i = 0; i < __n; ++i)
        std::cout << "input_after " << i << " " << in_debug2[i] << std::endl;
#endif

    //printf("KERNEL_TEMPLATE DONE %lu\n", __n);
}

} // inline namespace igpu

} // namespace oneapi::dpl::experimental::kt

#endif /* _ONEDPL_parallel_backend_sycl_scan_H */
