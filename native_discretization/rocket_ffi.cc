// JAX CPU FFI boundary. JAX owns the buffers; no Python callback or host copy.
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include "xla/ffi/api/ffi.h"
namespace ffi=xla::ffi;
extern "C" int icore_rocket_discretize(int,int,const double*,const double*,const double*,const double*,double*);
static ffi::Error RocketRk4(ffi::ThreadPool pool,ffi::BufferR2<ffi::F64> x,ffi::BufferR2<ffi::F64> u,
    ffi::BufferR1<ffi::F64> scales,ffi::BufferR1<ffi::F64> parameters,
    ffi::BufferR1<ffi::S32> config,ffi::BufferR0<ffi::F64> span,
    ffi::ResultBuffer<ffi::F64> result) {
    if(config.element_count()!=6) return ffi::Error::InvalidArgument("Rocket RK4 configuration must have six entries");
    const int32_t *c=config.typed_data();
    const int nodes=c[0],points=c[1],foh=c[2],free_time=c[3],history=c[4],nparam=c[5];
    if(nodes<2 || points<2 || (foh!=0&&foh!=1) || (free_time!=0&&free_time!=1) || (history!=0&&history!=1) ||
       x.dimensions()[0]!=nodes || x.dimensions()[1]!=15 || u.dimensions()[0]!=nodes || u.dimensions()[1]!=5 ||
       scales.element_count()!=nodes-1 || parameters.element_count()!=nparam || nparam!=68 ||
       result->element_count()!=int64_t(nodes-1)*420)
        return ffi::Error::InvalidArgument("Rocket RK4 buffer shape mismatch");
    for(int64_t i=0;i<x.element_count();++i) if(!std::isfinite(x.typed_data()[i])) return ffi::Error::InvalidArgument("Nonfinite state");
    for(int64_t i=0;i<u.element_count();++i) if(!std::isfinite(u.typed_data()[i])) return ffi::Error::InvalidArgument("Nonfinite input");
    for(int64_t i=0;i<parameters.element_count();++i) if(!std::isfinite(parameters.typed_data()[i])) return ffi::Error::InvalidArgument("Nonfinite model parameter");
    // Shooting intervals are independent. Reuse XLA's workers and wait before
    // returning, so buffers stay alive and the caller measures all RK4 work.
    // Avoid tiny tasks: aim for two intervals and 32 RK4 substeps per task.
    const int64_t groups=std::min<int64_t>(nodes/2,int64_t(nodes-1)*(points-1)/32);
    const int workers=std::max<int64_t>(1,std::min(groups,pool.num_threads()));
    auto run=[&](int worker) {
        const int begin=(nodes-1)*worker/workers, end=(nodes-1)*(worker+1)/workers;
        return icore_rocket_discretize(end-begin+1,points,x.typed_data()+15*begin,
            u.typed_data()+5*begin,scales.typed_data()+begin,parameters.typed_data(),
            result->typed_data()+420*begin);
    };
    int error=0;
    if(workers==1) {
        error=run(0);
    } else {
        std::mutex mutex;
        std::condition_variable done;
        int remaining=workers-1;
        for(int worker=1;worker<workers;++worker) pool.Schedule([&,worker]() {
            const int status=run(worker);
            std::lock_guard<std::mutex> lock(mutex);
            error|=status;
            if(--remaining==0) done.notify_one();
        });
        const int status=run(0);
        std::unique_lock<std::mutex> lock(mutex);
        done.wait(lock,[&] {return remaining==0;});
        error|=status;
    }
    if(error) return ffi::Error::InvalidArgument("Rocket RK4 rejected a nonfinite value or invalid duration");
    return ffi::Error::Success();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(ICORERocketRk4,RocketRk4,
    ffi::Ffi::Bind().Ctx<ffi::ThreadPool>().Arg<ffi::BufferR2<ffi::F64>>().Arg<ffi::BufferR2<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::F64>>().Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::S32>>().Arg<ffi::BufferR0<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>());
