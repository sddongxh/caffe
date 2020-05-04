#include <boost/thread.hpp>
#include <exception>
#include <caffe/caffe.hpp>

#include "caffe/internal_thread.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

InternalThread::InternalThread(int target_device, size_t rank, size_t threads, bool delayed,
    const std::string& name)
    : target_device_(target_device),
      rank_(rank),
      lwp_id_(0),
      lwp_id_parent_(caffe::lwp_id()),
      children_(threads),
      delay_flags_(threads, make_shared<Flag>(!delayed)),
      name_(name) {
    LOG(INFO) << "InternalThread " << lwp_id_parent_ << ": " << name;
}

void InternalThread::StartInternalThread(bool set_cpu_affinity, uint64_t random_seed) {
  CHECK(!is_started()) << "Threads should persist and not be restarted.";
  LOG(INFO) <<
#ifdef USE_MPI
    "{" << P2PManager::global_rank() << "} "
#endif
    "Starting " << children_.size() << " internal thread(s) on device " << target_device_;
  Caffe::Brew mode = Caffe::mode();
  if (mode == Caffe::GPU) {
    CHECK_GE(target_device_, 0);
  }
  try {
    for (size_t child_id = 0; child_id < children_.size(); ++child_id) {
        children_[child_id] = boost::thread(&InternalThread::entry, this, child_id,
                target_device_, mode, random_seed, rank_, set_cpu_affinity);
    }
  } catch (std::exception& e) {
    LOG(FATAL) << "Thread exception: " << e.what();
  }
}

void InternalThread::RestartAllThreads(size_t new_threads, bool delayed, bool set_cpu_affinity,
    uint64_t random_seed) {
  if (new_threads == 0UL) {
    return;
  }
  LOG(INFO) << "Restarting " << new_threads << " internal thread(s) on device " << target_device_;
  Caffe::Brew mode = Caffe::mode();
  if (mode == Caffe::GPU) {
    CHECK_GE(target_device_, 0);
  }
  children_.clear();
  delay_flags_.clear();
  children_.resize(new_threads);
  delay_flags_.resize(new_threads);
  try {
    for (size_t child_id = 0; child_id < new_threads; ++child_id) {
      CHECK(!is_started(child_id));
      delay_flags_[child_id] = make_shared<Flag>(!delayed);
      children_[child_id] = boost::thread(&InternalThread::entry, this, child_id,
          target_device_, mode, random_seed, rank_, set_cpu_affinity);
    }
  } catch (std::exception& e) {
    LOG(FATAL) << "Thread exception: " << e.what();
  }
}

void InternalThread::entry(size_t child_id, int device, Caffe::Brew mode, uint64_t random_seed,
    size_t rank, bool set_cpu_affinity) {
  lwp_id_ = caffe::lwp_id();
  delay_flags_[child_id]->wait();
  if (mode == Caffe::GPU) {
    CHECK_GE(device, 0);
  }
  rank_ = rank;
  target_device_ = device;

  if (mode == Caffe::GPU) {
    CUDA_CHECK(cudaSetDevice(device));
  }
  Caffe::set_mode(mode);
  Caffe::set_random_seed(random_seed);

  LOG(INFO) << "Started internal thread " << lwp_id()
            << " on device " << device << ", rank " << rank_;
  if (mode == Caffe::GPU && set_cpu_affinity) {
#ifndef NO_NVML
    nvml::setCpuAffinity(device);
#endif
  }
  if (children_.size() == 1) {
    InternalThreadEntry();
  } else {
    InternalThreadEntryN(child_id);
  }
}

void InternalThread::StopInternalThread(bool wait_all) {
  for (size_t child_id = 0; child_id < children_.size(); ++child_id) {
    if (is_started(child_id)) {
      children_[child_id].interrupt();
    }
  }
  if (wait_all) {
    WaitAll();
  }
}

void InternalThread::WaitAll() {
  try {
    for (size_t child_id = 0; child_id < children_.size(); ++child_id) {
      if (is_started(child_id)) {
        children_[child_id].join();
      }
    }
  } catch (boost::thread_interrupted&) {
  } catch (std::exception& e) {
    LOG(FATAL) << "Thread exception: " << e.what();
  }
}

}  // namespace caffe
