// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "src/worker.h"

#include "src/clock.h"
#include "src/profiler.h"
#include "src/symbols.h"
#include "src/throttler_api.h"
#include "src/throttler_timed.h"

DEFINE_bool(cprof_enabled, true,
            "when unset, unconditionally disable the profiling");
DEFINE_string(
    cprof_profile_filename, "",
    "when set to a path, store profiles locally at the specified prefix");
DEFINE_int32(cprof_cpu_sampling_period_msec, 10,
             "sampling period for CPU time profiling, in milliseconds");
DEFINE_int32(cprof_wall_sampling_period_msec, 100,
             "sampling period for wall time profiling, in milliseconds");

namespace cloud {
namespace profiler {

std::atomic<bool> Worker::enabled_;

Worker* Worker::instance = NULL;

void Worker::Start(JNIEnv *jni) {
  jclass cls = jni->FindClass("java/lang/Thread");
  jmethodID constructor = jni->GetMethodID(cls, "<init>", "()V");
  jobject thread = jni->NewGlobalRef(jni->NewObject(cls, constructor));
  if (thread == nullptr) {
    LOG(ERROR) << "Failed to construct cloud profiler worker thread";
    return;
  }

  // Pass 'this' as the arg to access members from the worker thread.
  jvmtiError err = jvmti_->RunAgentThread(thread, ProfileThread, this,
                                          JVMTI_THREAD_MIN_PRIORITY);
  if (err) {
    LOG(ERROR) << "Failed to start cloud profiler worker thread";
    return;
  }

  enabled_ = FLAGS_cprof_enabled;
}

void Worker::Stop() {
  // Signal the worker thread to exit and wait until it does.
  stopping_.store(true, std::memory_order_release);
  std::lock_guard<std::mutex> lock(mutex_);
}

namespace {

string Collect(Profiler *p,
               google::javaprofiler::NativeProcessInfo *native_info) {
  const char *profile_type = p->ProfileType();
  if (!p->Collect()) {
    LOG(ERROR) << "Failure: Could not collect " << profile_type << " profile";
    return "";
  }
  native_info->Refresh();
  return p->SerializeProfile(*native_info);
}

}  // namespace

void Worker::EnableProfiling() {
  enabled_ = true;
}

void Worker::DisableProfiling() {
  enabled_ = false;
}

void Worker::ProfileThread(jvmtiEnv *jvmti_env, JNIEnv *jni_env, void *arg) {
  Worker *w = static_cast<Worker *>(arg);
  google::javaprofiler::NativeProcessInfo n("/proc/self/maps");

  w->resetSymbols();

  std::unique_ptr<Throttler> t =
      FLAGS_cprof_profile_filename.empty()
          ? std::unique_ptr<Throttler>(new APIThrottler())
          : std::unique_ptr<Throttler>(
                new TimedThrottler(FLAGS_cprof_profile_filename));

  while (t->WaitNext()) {
    std::lock_guard<std::mutex> lock(w->mutex_);
    if (w->stopping_) {
      // The worker is exiting.
      break;
    }
    if (!enabled_) {
      // Skip the collection and upload steps when profiling is disabled.
      continue;
    }
    if (w->_native_lib_refresh++ % 10 == 0) {
      LOG(INFO) << "refreshing symbol map";
      w->resetSymbols();
    }

    string profile;
    string pt = t->ProfileType();
    if (pt == kTypeCPU) {
      CPUProfiler p(w->jvmti_, w->threads_, t->DurationNanos(),
                    FLAGS_cprof_cpu_sampling_period_msec * kNanosPerMilli);
      profile = Collect(&p, &n);
    } else if (pt == kTypeWall) {
      // Note that the requested sampling period for the wall profiling may be
      // increased if the number of live threads is too large.
      WallProfiler p(w->jvmti_, w->threads_, t->DurationNanos(),
                     FLAGS_cprof_wall_sampling_period_msec * kNanosPerMilli);
      profile = Collect(&p, &n);
    } else {
      LOG(ERROR) << "Unknown profile type '" << pt << "', skipping the upload";
      continue;
    }
    if (profile.empty()) {
      LOG(ERROR) << "No profile bytes collected, skipping the upload";
      continue;
    }
    if (!t->Upload(profile)) {
      LOG(ERROR) << "Error on profile upload, discarding the profile";
    }
  }
  LOG(INFO) << "Exiting the profiling loop";
}

void Worker::addJavaMethod(const void* address, int length, jmethodID method) {
    _jit_lock.lock();
    _java_methods.add(address, length, method);
    updateJitRange(address, (const char*)address + length);
    _jit_lock.unlock();
}

void Worker::removeJavaMethod(const void* address, jmethodID method) {
    _jit_lock.lock();
    _java_methods.remove(address, method);
    _jit_lock.unlock();
}

void Worker::addRuntimeStub(const void* address, int length, const char* name) {
    _jit_lock.lock();
    _runtime_stubs.add(address, length, name);
    updateJitRange(address, (const char*)address + length);
    _jit_lock.unlock();
}

void Worker::updateJitRange(const void* min_address, const void* max_address) {
    if (min_address < _jit_min_address) _jit_min_address = min_address;
    if (max_address > _jit_max_address) _jit_max_address = max_address;
}

bool Worker::addressInCode(const void* pc) {
    // 1. Check if PC lies within JVM's compiled code cache
    // Address in CodeCache is executable if it belongs to a Java method or a runtime stub
    if (pc >= _jit_min_address && pc < _jit_max_address) {
        _jit_lock.lockShared();
        bool valid = _java_methods.find(pc) != NULL || _runtime_stubs.find(pc) != NULL;
        _jit_lock.unlockShared();
        return valid;
    }

    // 2. Check if PC belongs to executable code of shared libraries
    for (int i = 0; i < _native_lib_count; i++) {
        if (_native_libs[i]->contains(pc)) {
            return true;
        }
    }

    // This can be some other dynamically generated code, but we don't know it. Better stay safe.
    return false;
}

void Worker::resetSymbols() {
    for (int i = 0; i < _native_lib_count; i++) {
        delete _native_libs[i];
    }
    _native_lib_count = Symbols::parseMaps(_native_libs, MAX_NATIVE_LIBS);
    NativeCodeCache *libjvm = jvmLibrary();
    if (libjvm != NULL) {
      initJvmtiFunctions(libjvm);
    }
}

void Worker::initJvmtiFunctions(NativeCodeCache* libjvm) {
    if (_JvmtiEnv_GetStackTrace == NULL) {
        // Find ThreadLocalStorage::thread() if exists
        if (_ThreadLocalStorage_thread == NULL) {
            _ThreadLocalStorage_thread = (void* (*)()) libjvm->findSymbol("_ZN18ThreadLocalStorage6threadEv");
        }
        // Fallback to ThreadLocalStorage::get_thread_slow()
        if (_ThreadLocalStorage_thread == NULL) {
            _ThreadLocalStorage_thread = (void* (*)()) libjvm->findSymbol("_ZN18ThreadLocalStorage15get_thread_slowEv");
        }
        // Fallback to Thread::current(), e.g. on Zing
        if (_ThreadLocalStorage_thread == NULL) {
            _ThreadLocalStorage_thread = (void* (*)()) libjvm->findSymbol("_ZN6Thread7currentEv");
        }
        // JvmtiEnv::GetStackTrace(JavaThread* java_thread, jint start_depth, jint max_frame_count, jvmtiFrameInfo* frame_buffer, jint* count_ptr)
        if (_ThreadLocalStorage_thread != NULL) {
            _JvmtiEnv_GetStackTrace = (jvmtiError (*)(void*, void*, jint, jint, jvmtiFrameInfo*, jint*))
                libjvm->findSymbol("_ZN8JvmtiEnv13GetStackTraceEP10JavaThreadiiP15_jvmtiFrameInfoPi");
        }

        if (_JvmtiEnv_GetStackTrace == NULL) {
            fprintf(stderr, "WARNING: Install JVM debug symbols to improve profile accuracy\n");
        }
    }
}

NativeCodeCache* Worker::jvmLibrary() {
    const void* asgct = (const void*)google::javaprofiler::Asgct::GetAsgct();

    for (int i = 0; i < _native_lib_count; i++) {
        if (_native_libs[i]->contains(asgct)) {
            return _native_libs[i];
        }
    }
    return NULL;
}

const void* Worker::findSymbol(const char* name) {
    for (int i = 0; i < _native_lib_count; i++) {
        const void* address = _native_libs[i]->findSymbol(name);
        if (address != NULL) {
            return address;
        }
    }
    return NULL;
}


}  // namespace profiler
}  // namespace cloud
