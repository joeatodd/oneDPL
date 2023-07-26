#include <execution>
#include <sycl/sycl.hpp>

#define _ONEDPL_PREDEFINED_POLICIES 0

#include "parallel_backend_sycl_reduce.h"

using namespace oneapi::dpl::__par_backend_hetero;

template <typename T>
struct noop {
    inline __attribute__((always_inline)) T operator() (T input) const { return input; }

    template <typename size, typename acc>
    inline __attribute__((always_inline)) T operator() (size idx, acc data) const { return data[idx]; }
};

template <typename T>
std::vector<T> generate_input(size_t n) {
    std::vector<T> data(n);
    for (int i = 5; i < data.size(); ++i)
        data[i] = i;
    return data;
}

int main() {

    sycl::queue Q;

    __parallel_reduce_sub_group_submitter<32, 64, int> my_reduce;
    for (size_t n = 16; n < 32 * 64; n += 25) {
      int init = 0;
  
      // __parallel_reduce_single_work_group_submitter<256, 32, 4, int> my_reduce;
      // __parallel_reduce_multi_work_group_submitter<256, 32, 4, int> my_reduce;
  
      auto in_data = generate_input<int>(n);
      auto expected = std::reduce(in_data.begin(), in_data.end());
  
      sycl::buffer<int> in(in_data);
      sycl::buffer<int> out(1);
  
      my_reduce(Q, n, sycl::plus<int>(), noop<int>(), init, in, out);
      Q.wait();
  
      sycl::host_accessor host_acc(out);
      std::cout << "n:" << n << " result:" << host_acc[0] << " expected:" << expected << std::endl;
      if (host_acc[0] != expected)
          std::cout << "FAILED!\n";
      // assert(host_acc[0] == expected);
    }


    return 0;
}
