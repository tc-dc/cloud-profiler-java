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

#ifndef THIRD_PARTY_JAVAPROFILER_PROFILE_PROTO_BUILDER_H__
#define THIRD_PARTY_JAVAPROFILER_PROFILE_PROTO_BUILDER_H__

#include <link.h>
#include <memory>
#include <string>
#include <vector>

#include "perftools/profiles/proto/builder.h"
#include "third_party/java/jdk/include/jvmti.h"
#include "third_party/javaprofiler/stacktrace_decls.h"

namespace google {
namespace javaprofiler {

struct ProfileStackTrace {
  JVMPI_CallTrace *trace;
  jint metric_value;
};

// Store proto sample objects for specific stack traces.
class TraceSamples {
 public:
  perftools::profiles::Sample *SampleFor(const JVMPI_CallTrace &trace) const;
  void Add(const JVMPI_CallTrace &trace, perftools::profiles::Sample *sample);

 private:
  struct TraceHash {
    size_t operator()(const JVMPI_CallTrace &trace) const;
  };

  struct TraceEquals {
    bool operator()(const JVMPI_CallTrace &trace1,
                    const JVMPI_CallTrace &trace2) const;
  };

  __gnu_cxx::hash_map<JVMPI_CallTrace, perftools::profiles::Sample *, TraceHash,
                      TraceEquals>
      traces_;
};

// Store locations previously seen so that the profile is only
// modified for new locations.
class LocationBuilder {
 public:
  explicit LocationBuilder(perftools::profiles::Builder *builder)
      : builder_(builder) {}

  // Return an existing or new location matching the given parameters,
  // modifying the profile as needed to add new function and location
  // information.
  perftools::profiles::Location *LocationFor(const string &class_name,
                                             const string &function_name,
                                             const string &file_name,
                                             int line_number);

 private:
  struct LocationInfo {
    string class_name;
    string function_name;
    string file_name;
    int line_number;
  };

  struct LocationInfoHash {
    size_t operator()(const LocationInfo &info) const;
  };

  struct LocationInfoEquals {
    bool operator()(const LocationInfo &info1, const LocationInfo &info2) const;
  };

  perftools::profiles::Builder *builder_;

  __gnu_cxx::hash_map<LocationInfo, perftools::profiles::Location *,
                      LocationInfoHash, LocationInfoEquals>
      locations_;
};

// Remember traces and use the information to create locations with native
// information if supported.
class ProfileFrameCache {
 public:
  virtual void ProcessTraces(const ProfileStackTrace *traces,
                             int num_traces) = 0;

  virtual perftools::profiles::Location *GetLocation(
      const JVMPI_CallFrame &jvm_frame, LocationBuilder *location_builder) = 0;

  virtual string GetFunctionName(const JVMPI_CallFrame &jvm_frame) = 0;

  virtual ~ProfileFrameCache() {}
};

// Create profile protobufs from traces obtained from JVM profiling.
class ProfileProtoBuilder {
 public:
  virtual ~ProfileProtoBuilder() {}

  // Add traces to the proto.
  void AddTraces(const ProfileStackTrace *traces, int num_traces);

  // Add traces to the proto, where each trace has a defined count
  // of occurrences.
  void AddTraces(const ProfileStackTrace *traces,
                 const int32 *counts,
                 int num_traces);

  // Add a "fake" trace with a single frame. Used to represent JVM
  // tasks such as JIT compilation and GC.
  void AddArtificialTrace(const string& name, int count, int sampling_rate);

  // Build the proto. Calling any other method on the class after calling
  // this has undefined behavior.
  virtual std::unique_ptr<perftools::profiles::Profile> CreateProto() = 0;

  static std::unique_ptr<ProfileProtoBuilder> ForHeap(
      jvmtiEnv *jvmti_env, int64 sampling_rate, ProfileFrameCache *cache);

  static std::unique_ptr<ProfileProtoBuilder> ForCpu(
      jvmtiEnv *jvmti_env, int64 sampling_rate, ProfileFrameCache *cache);

  static std::unique_ptr<ProfileProtoBuilder> ForContention(
      jvmtiEnv *jvmti_env, int64 sampling_rate, ProfileFrameCache *cache);

 protected:
  struct SampleType {
    SampleType(const string &type_in, const string &unit_in)
        : type(type_in), unit(unit_in) {}

    string type;
    string unit;
  };

  ProfileProtoBuilder(jvmtiEnv *jvmti_env,
                      ProfileFrameCache *native_cache,
                      int64 sampling_rate,
                      const SampleType &count_type,
                      const SampleType &metric_type);

  // An implementation must decide how many frames to skip in a trace.
  virtual int SkipTopNativeFrames(const JVMPI_CallTrace &trace) = 0;

  // Build the proto, unsampling the sample metrics. Calling any other method
  // on the class after calling this has undefined behavior.
  std::unique_ptr<perftools::profiles::Profile> CreateUnsampledProto();

  // Build the proto, without normalizing the sampled metrics. Calling any
  // other method on the class after calling this has undefined behavior.
  std::unique_ptr<perftools::profiles::Profile> CreateSampledProto();

  perftools::profiles::Builder builder_;

 private:
  // Track progress through a stack as we traverse it, in order to determine
  // how processing should proceed based on the context of a frame.
  class StackState {
   public:
    StackState() {
    }

    // Notify the state that we are visiting a Java frame.
    void JavaFrame() {
      in_jni_helpers_ = false;
    }

    // Notify the state that we are visiting a native frame.
    void NativeFrame(const string &function_name) {
      if (StartsWith(function_name, "JavaCalls::call_helper")) {
        in_jni_helpers_ = true;
      }
    }

    // Should we skip the current frame?
    bool SkipFrame() const {
      return in_jni_helpers_;
    }

   private:
    // We don't add native frames that are just "helper" native code for
    // dispatching to JNI code. We determine this by detecting a native
    // JavaCalls::call_helper frame, then skipping until the we see a Java
    // frame again.
    // TODO: Support a "complete detail" mode to override this.
    bool in_jni_helpers_ = false;

    static bool StartsWith(const string &s, const string &prefix) {
      return s.find(prefix) == 0;
    }
  };

  void AddSampleType(const SampleType &sample_type);
  void SetPeriodType(const SampleType &metric_type);
  void InitSampleValues(perftools::profiles::Sample *sample, jint metric);
  void InitSampleValues(perftools::profiles::Sample *sample, jint count,
                        jint metric);
  void UpdateSampleValues(perftools::profiles::Sample *sample, jint count,
                          jint size);
  void AddTrace(const ProfileStackTrace &trace, int32 count);
  void AddJavaInfo(const google::javaprofiler::JVMPI_CallFrame &jvm_frame,
                   perftools::profiles::Profile *profile,
                   perftools::profiles::Sample *sample,
                   StackState *stack_state);
  void AddNativeInfo(const google::javaprofiler::JVMPI_CallFrame &jvm_frame,
                     perftools::profiles::Profile *profile,
                     perftools::profiles::Sample *sample,
                     StackState *stack_state);
  void UnsampleMetrics();

  jvmtiEnv *jvmti_env_;

  ProfileFrameCache *native_cache_;
  TraceSamples trace_samples_;
  LocationBuilder location_builder_;
  int64 sampling_rate_ = 0;
};

// Computes the ratio to use to scale heap data to unsample it.
// Accounts for the probability of it appearing in the
// collected data based on exponential samples. heap profiles rely
// on a poisson process to determine which samples to collect, based
// on the desired average collection rate R. The probability of a
// sample of size S to appear in that profile is 1-exp(-S/R).
double CalculateSamplingRatio(int64 rate, int64 count, int64 metric_value);

class CpuProfileProtoBuilder : public ProfileProtoBuilder {
 public:
  CpuProfileProtoBuilder(jvmtiEnv *jvmti_env,
                         int64 sampling_rate,
                         ProfileFrameCache *cache)
      : ProfileProtoBuilder(jvmti_env, cache, sampling_rate,
                            ProfileProtoBuilder::SampleType("samples", "count"),
                            ProfileProtoBuilder::SampleType("cpu",
                                                            "nanoseconds")) {
    builder_.mutable_profile()->set_period(sampling_rate);
  }

  std::unique_ptr<perftools::profiles::Profile> CreateProto() override {
    return CreateSampledProto();
  }

 protected:
  int SkipTopNativeFrames(const JVMPI_CallTrace &trace) override { return 0; }
};

class HeapProfileProtoBuilder : public ProfileProtoBuilder {
 public:
  HeapProfileProtoBuilder(jvmtiEnv *jvmti_env,
                          int64 sampling_rate,
                          ProfileFrameCache *cache)
      : ProfileProtoBuilder(jvmti_env, cache, sampling_rate,
                            ProfileProtoBuilder::SampleType("inuse_objects",
                                                            "count"),
                            ProfileProtoBuilder::SampleType("inuse_space",
                                                            "bytes")) {
  }

  std::unique_ptr<perftools::profiles::Profile> CreateProto() override {
    return CreateUnsampledProto();
  }

 protected:
  int SkipTopNativeFrames(const JVMPI_CallTrace &trace) override {
    for (int i = 0; i < trace.num_frames; ++i) {
      if (trace.frames[i].lineno !=
          google::javaprofiler::kNativeFrameLineNum) {
        return i;
      }
    }

    return trace.num_frames;
  }
};

class ContentionProfileProtoBuilder : public ProfileProtoBuilder {
 public:
  ContentionProfileProtoBuilder(jvmtiEnv *jvmti_env,
                                int64 sampling_rate,
                                ProfileFrameCache *cache)
      : ProfileProtoBuilder(jvmti_env, cache, sampling_rate,
                            ProfileProtoBuilder::SampleType("contentions",
                                                            "count"),
                            ProfileProtoBuilder::SampleType("delay",
                                                            "microseconds")) {
    builder_.mutable_profile()->set_period(sampling_rate);
  }

  std::unique_ptr<perftools::profiles::Profile> CreateProto() {
    return CreateSampledProto();
  }

 protected:
  int SkipTopNativeFrames(const JVMPI_CallTrace &trace) override { return 0; }
};

}  // namespace javaprofiler
}  // namespace google

#endif  // THIRD_PARTY_JAVAPROFILER_PROFILE_PROTO_BUILDER_H__
