/*
 * Copyright 2018 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CLOUD_PROFILER_AGENT_JAVA_WORKER_H_
#define CLOUD_PROFILER_AGENT_JAVA_WORKER_H_

#include <atomic>
#include <mutex>  // NOLINT

#include "src/globals.h"
#include "src/threads.h"
#include "src/throttler.h"
#include "src/spinLock.h"
#include "src/codeCache.h"

namespace cloud {
namespace profiler {

const int MAX_NATIVE_LIBS   = 2048;


class Worker {
 public:
  Worker(jvmtiEnv *jvmti, ThreadTable *threads)
      : _ThreadLocalStorage_thread(NULL),
        _JvmtiEnv_GetStackTrace(NULL),
        _jit_lock(),
        _jit_min_address((const void*)-1),
        _jit_max_address((const void*)0),
        _java_methods(),
        _runtime_stubs("[stubs]"),
        _native_lib_count(0),
        _native_lib_refresh(0),
        jvmti_(jvmti), 
        threads_(threads), 
        stopping_() {}

  void Start(JNIEnv *jni);
  void Stop();

  static void EnableProfiling();
  static void DisableProfiling();

  static Worker* instance;

  void addJavaMethod(const void* address, int length, jmethodID method);
  void removeJavaMethod(const void* address, jmethodID method);
  void addRuntimeStub(const void* address, int length, const char* name);

  const void* getJitMin() { return _jit_min_address; }
  const void* getJitMax() { return _jit_max_address; }
  CodeCache *getCodeCache() { return &_java_methods; }
  NativeCodeCache *getNativeCodeCache() { return &_runtime_stubs; }

  void jitLockShared() { _jit_lock.lockShared(); }
  void jitUnlockShared() { _jit_lock.unlockShared(); }

  bool addressInCode(const void* pc);
  void resetSymbols();


  void* (*_ThreadLocalStorage_thread)();
  jvmtiError (*_JvmtiEnv_GetStackTrace)(void* self, void* thread, jint start_depth, jint max_frame_count,
                                      jvmtiFrameInfo* frame_buffer, jint* count_ptr);
  
 private:
  static void ProfileThread(jvmtiEnv *jvmti_env, JNIEnv *jni_env, void *arg);
  void updateJitRange(const void* min_address, const void* max_address);

  void initJvmtiFunctions(NativeCodeCache* libjvm);
  NativeCodeCache* jvmLibrary();
  const void* findSymbol(const char* name);

  SpinLock _jit_lock;
  const void* _jit_min_address;
  const void* _jit_max_address;
  CodeCache _java_methods;
  NativeCodeCache _runtime_stubs;
  NativeCodeCache* _native_libs[MAX_NATIVE_LIBS];
  int _native_lib_count;
  int _native_lib_refresh;

  jvmtiEnv *jvmti_;
  ThreadTable *threads_;
  std::mutex mutex_;  // Held by the worker thread while it's running.
  std::atomic<bool> stopping_;
  static std::atomic<bool> enabled_;
  DISALLOW_COPY_AND_ASSIGN(Worker);
};

}  // namespace profiler
}  // namespace cloud

#endif  // CLOUD_PROFILER_AGENT_JAVA_WORKER_H_
