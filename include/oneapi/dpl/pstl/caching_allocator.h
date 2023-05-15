#ifndef _ONEDPL_CACHING_ALLOCATOR_H
#define _ONEDPL_CACHING_ALLOCATOR_H

#include <sycl/sycl.hpp>
#include <mutex>
#include <map>

// TODO: see about storing queue and adding default constructor
class mem_block
{
  public:
    mem_block(const sycl::queue& queue, size_t bytes) : bytes_(bytes), is_free_(true)
    {
        d_ptr_ = sycl::malloc_device(bytes_, queue);
    };

    // TODO: add safety checks, i.e. cant return memory already returned.
    // can't request memory that is already free
    void
    return_mem()
    {
        is_free_ = true;
    }
    void*
    request_mem()
    {
        is_free_ = false;
        return d_ptr_;
    }

    inline size_t
    get_bytes() const
    {
        return bytes_;
    }
    inline bool
    is_free() const
    {
        return is_free_;
    }
    inline bool
    is_same(const void* d_ptr) const
    {
        return d_ptr == d_ptr_;
    }

    void
    free_mem_block(const sycl::queue& queue)
    {
        sycl::free(d_ptr_, queue);
        bytes_ = 0;
        is_free_ = false;
    }

  private:
    size_t bytes_;
    bool is_free_;
    void* d_ptr_;
};

// TODO: consider rewriting map with custom mem_block class
class mem_block_manager
{
  public:
    mem_block_manager(){};

    void*
    get_mem(const sycl::queue& queue, size_t bytes)
    {
        // Search through mem blocks for a valid block
        for (unsigned int i = 0; i < queues_.size(); ++i)
        {
            if (queues_[i] == queue)
            {
                for (auto& block : mem_blocks_[i])
                {
                    if (block.is_free() && block.get_bytes() <= bytes)
                    {
                        return block.request_mem();
                    }
                }
            }
        }

        // If none available create new mem_block
        for (unsigned int i = 0; i < queues_.size(); ++i)
        {
            if (queues_[i] == queue)
            {
                mem_blocks_[i].push_back(mem_block(queue, bytes));
                return mem_blocks_[i].back().request_mem();
            }
        }

        // New queue
        queues_.push_back(queue);
        mem_blocks_.push_back(std::vector<mem_block>{mem_block(queue, bytes)});
        return mem_blocks_.back().back().request_mem();
    }

    void
    return_mem(const sycl::queue& queue, void* d_ptr)
    {
        for (unsigned int i = 0; i < queues_.size(); ++i)
        {
            if (queues_[i] == queue)
            {
                for (auto& block : mem_blocks_[i])
                {
                    if (block.is_same(d_ptr))
                    {
                        block.return_mem();
                    }
                }
            }
        }

        // Not 100% sure how to handle case where memory is being freed but not created by CachingAllocator
        // Currently just gonna manually free it
        sycl::free(d_ptr, queue);
    }

    void
    release_all_memory()
    {
        for (unsigned int i = 0; i < queues_.size(); ++i)
            for (auto& block : mem_blocks_[i])
                block.free_mem_block(queues_[i]);
    }

  private:
    // Naive implementation
    std::vector<sycl::queue> queues_;
    std::vector<std::vector<mem_block>> mem_blocks_;
};

// Class relies on mutex locks to protect memory and make it thread safe
class CachingDeviceAllocator
{
  public:
    CachingDeviceAllocator() {};

    void
    DeviceAllocate(const sycl::queue& queue, void** d_ptr, size_t bytes)
    {
        std::lock_guard<std::mutex> guard(memory_manager_mutex);
        *d_ptr = memory_manager.get_mem(queue, bytes);
    }

    template <typename ptrT>
    void
    DeviceAllocate(const sycl::queue& queue, ptrT** d_ptr, size_t size)
    {
        DeviceAllocate(queue, reinterpret_cast<void**>(d_ptr), size*sizeof(ptrT));
    }

    void
    DeviceFree(const sycl::queue& queue, void* d_ptr)
    {
        std::lock_guard<std::mutex> guard(memory_manager_mutex);
        memory_manager.return_mem(queue, d_ptr);
    }

    void
    FreeAllCached()
    {
        std::lock_guard<std::mutex> guard(memory_manager_mutex);
        memory_manager.release_all_memory();
    }

  private:
    // Map stores a vector which contains a vector of mem_blocks
    mem_block_manager memory_manager;
    std::mutex memory_manager_mutex;
};

#endif // _ONEDPL_CACHING_ALLOCATOR_H
