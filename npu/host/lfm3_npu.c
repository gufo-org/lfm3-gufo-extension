// liblfm3_npu: in-process Loom compilation and dispatch on the XDNA NPU.
//
// Follows loom-run's flow (admit source, prepare the default pass program,
// select the live HAL target, compile one root, prepare the executable) and
// keeps the runtime, compiler and device buffers alive across calls. Host data
// moves through the runtime's mappable staging pool and queue transfers.

#include "lfm3_npu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "loom/tooling/cli/loomc_options.h"
#include "loom/tooling/cli/loomc_result.h"
#include "loom/tooling/execution/hal/invocation.h"
#include "loom/tooling/execution/hal/runtime.h"
#include "loom/tooling/input/loomc.h"
#include "loom/tooling/input/loomc_configured.h"
#include "iree/tooling/value_io.h"
#include "loomc/compile.h"
#include "loomc/launch_config.h"
#include "loomc/iree.h"
#include "loomc/target/amd/xdna/iree_hal.h"
#include "loomc/target/configured.h"
#include "loomc/target/iree_hal.h"

#define LFM3_NPU_MAX_KERNELS 256
#define LFM3_NPU_MAX_BUFFERS 4096

typedef struct lfm3_npu_kernel_t {
  loom_run_hal_prepared_candidate_t candidate;
  iree_hal_executable_function_t function;
  char export_name[256];
  uint32_t workgroup_count[3];
} lfm3_npu_kernel_t;

struct lfm3_npu {
  iree_allocator_t allocator;
  iree_arena_block_pool_t block_pool;
  loomc_target_environment_t* target_environment;
  loomc_context_t* context;
  loomc_workspace_t* workspace;
  loomc_compiler_t* compiler;
  loom_run_hal_runtime_t runtime;
  bool runtime_ready;
  loomc_target_profile_t* target_profile;
  const iree_hal_executable_target_t* executable_target;
  lfm3_npu_kernel_t* kernels[LFM3_NPU_MAX_KERNELS];
  int kernel_count;
  iree_hal_buffer_t* buffers[LFM3_NPU_MAX_BUFFERS];
  int buffer_count;
  // Dispatches chain on one timeline: each waits for the previous value.
  iree_hal_semaphore_t* timeline;
  uint64_t submitted;
  char error[4096];
};

static int lfm3_npu_fail(lfm3_npu* npu, iree_status_t status) {
  if (iree_status_is_ok(status)) return 0;
  if (npu) {
    iree_host_size_t length = 0;
    char* text = NULL;
    if (iree_status_to_string(status, &npu->allocator, &text, &length)) {
      snprintf(npu->error, sizeof(npu->error), "%.*s", (int)length, text);
      iree_allocator_free(npu->allocator, text);
    } else {
      snprintf(npu->error, sizeof(npu->error), "status code %d",
               (int)iree_status_code(status));
    }
  }
  iree_status_free(status);
  return 1;
}

// Captures a loomc result's diagnostics into npu->error; returns true when the
// operation succeeded.
static iree_status_t lfm3_npu_check_result(lfm3_npu* npu, loomc_result_t* result,
                                           const char* what) {
  if (result == NULL) return iree_ok_status();
  char* text = NULL;
  size_t length = 0;
  FILE* stream = open_memstream(&text, &length);
  bool succeeded = false;
  iree_status_t status = stream ? loom_tooling_cli_print_loomc_result(
                                      stream, result, &succeeded)
                                : iree_make_status(IREE_STATUS_INTERNAL,
                                                   "open_memstream failed");
  if (stream) fclose(stream);
  loomc_result_release(result);
  if (iree_status_is_ok(status) && !succeeded) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s failed:\n%.*s",
                              what, (int)iree_min(length, 3500), text ? text : "");
  }
  free(text);
  return status;
}

int lfm3_npu_abi_version(void) { return LFM3_NPU_ABI_VERSION; }

const char* lfm3_npu_error(const lfm3_npu* npu) {
  return npu ? npu->error : "no NPU context";
}

int lfm3_npu_open(const char* device, lfm3_npu** out) {
  *out = NULL;
  lfm3_npu* npu = (lfm3_npu*)calloc(1, sizeof(*npu));
  if (!npu) return 1;
  npu->allocator = iree_allocator_system();
  iree_arena_block_pool_initialize(32 * 1024, npu->allocator, &npu->block_pool);
  *out = npu;

  // The HAL device comes from the --device flag, as in the Loom tools.
  char device_flag[512];
  snprintf(device_flag, sizeof(device_flag), "--device=%s", device);
  char program[] = "lfm3_npu";
  char* argv[] = {program, device_flag, NULL};
  int argc = 2;
  char** argv_ptr = argv;
  iree_status_t status = iree_flags_parse(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc,
                                          &argv_ptr);
  iree_string_view_t driver = iree_string_view_empty();
  iree_string_view_t rest = iree_string_view_empty();
  iree_string_view_split(iree_make_cstring_view(device), ':', &driver, &rest);

  const loomc_allocator_t loom_allocator =
      loomc_allocator_from_iree(npu->allocator);
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_target_environment_create_configured(
        loom_allocator, &npu->target_environment));
  }
  if (iree_status_is_ok(status)) {
    const loomc_context_target_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        .structure_size = sizeof(target_options),
        .target_environment = npu->target_environment,
    };
    const loomc_context_options_t context_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
        .structure_size = sizeof(context_options),
        .next = &target_options,
    };
    status = iree_status_from_loomc(
        loomc_context_create(&context_options, loom_allocator, &npu->context));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(
        loomc_workspace_create(NULL, loom_allocator, &npu->workspace));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_compiler_create(
        npu->context, NULL, loom_allocator, &npu->compiler));
  }
  if (iree_status_is_ok(status)) {
    loom_run_hal_runtime_options_t runtime_options;
    loom_run_hal_runtime_options_initialize(driver, &runtime_options);
    status = loom_run_hal_runtime_initialize(&runtime_options, npu->allocator,
                                             &npu->runtime);
    npu->runtime_ready = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(npu->runtime.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                       IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &npu->timeline);
  }
  if (iree_status_is_ok(status)) {
    const loomc_iree_hal_target_provider_t* providers[] = {
        loomc_xdna_iree_hal_target_provider(),
    };
    const loomc_iree_hal_target_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_IREE_HAL_TARGET_OPTIONS,
        .structure_size = sizeof(target_options),
        .identifier = loomc_make_cstring_view("lfm3 npu"),
        .device = npu->runtime.device,
        .physical_device_affinity =
            loom_run_hal_runtime_dispatch_physical_device_affinity(
                &npu->runtime),
        .target_profile = NULL,
        .providers = providers,
        .provider_count = 1,
    };
    loomc_iree_hal_target_selection_t selection = {0};
    loomc_result_t* result = NULL;
    status = iree_status_from_loomc(loomc_target_select_iree_hal(
        npu->target_environment, &target_options, loom_allocator, &selection,
        &result));
    if (iree_status_is_ok(status)) {
      status = lfm3_npu_check_result(npu, result, "target selection");
    } else {
      loomc_result_release(result);
    }
    npu->target_profile = selection.target_profile;
    npu->executable_target = selection.executable_target;
  }
  return lfm3_npu_fail(npu, status);
}

void lfm3_npu_close(lfm3_npu* npu) {
  if (!npu) return;
  for (int i = 0; i < npu->buffer_count; ++i) {
    iree_hal_buffer_release(npu->buffers[i]);
  }
  for (int i = 0; i < npu->kernel_count; ++i) {
    if (!npu->kernels[i]) continue;
    loom_run_hal_prepared_candidate_deinitialize(&npu->kernels[i]->candidate);
    free(npu->kernels[i]);
  }
  if (npu->timeline) {
    iree_status_ignore(loom_run_hal_semaphore_wait(npu->timeline, npu->submitted, iree_infinite_timeout(),
                                                   IREE_ASYNC_WAIT_FLAG_NONE));
    iree_hal_semaphore_release(npu->timeline);
  }
  if (npu->runtime_ready) loom_run_hal_runtime_deinitialize(&npu->runtime);
  loomc_target_profile_release(npu->target_profile);
  loomc_compiler_release(npu->compiler);
  loomc_workspace_release(npu->workspace);
  loomc_context_release(npu->context);
  loomc_target_environment_release(npu->target_environment);
  iree_arena_block_pool_deinitialize(&npu->block_pool);
  free(npu);
}

static iree_status_t lfm3_npu_launch_config(const loomc_artifact_t* artifact,
                                            iree_string_view_t export_name,
                                            iree_allocator_t allocator,
                                            uint32_t workgroup_count[3]) {
  workgroup_count[0] = workgroup_count[1] = workgroup_count[2] = 1;
  if (artifact == NULL) return iree_ok_status();
  loomc_launch_config_program_t* program = NULL;
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_launch_config_program_load(
      artifact, loomc_allocator_from_iree(allocator), &program)));
  loomc_launch_config_function_t function =
      loomc_launch_config_function_invalid();
  iree_status_t status =
      iree_status_from_loomc(loomc_launch_config_program_lookup_function(
          program, loomc_string_view_from_iree(export_name), &function));
  if (iree_status_is_ok(status)) {
    loomc_launch_config_t config = {
        .type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
        .structure_size = sizeof(config),
    };
    status = iree_status_from_loomc(loomc_launch_config_program_invoke(
        program, function, NULL, 0, &config));
    if (iree_status_is_ok(status)) {
      workgroup_count[0] = config.workgroup_count.x;
      workgroup_count[1] = config.workgroup_count.y;
      workgroup_count[2] = config.workgroup_count.z;
    }
  }
  loomc_launch_config_program_release(program);
  return status;
}

int lfm3_npu_compile(lfm3_npu* npu, const char* source, size_t source_length,
                     const char* source_name, const char* root,
                     int* out_kernel) {
  *out_kernel = -1;
  if (npu->kernel_count >= LFM3_NPU_MAX_KERNELS) {
    return lfm3_npu_fail(npu, iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                               "too many kernels"));
  }
  const loomc_allocator_t loom_allocator =
      loomc_allocator_from_iree(npu->allocator);
  loomc_module_t* module = NULL;
  loomc_pass_program_t* pass_program = NULL;
  loomc_result_t* result = NULL;
  loom_tooling_loomc_input_options_t admission = {
      .input = {.format = IREE_SV("loom")},
      .path = iree_make_cstring_view(source_name),
      .source = iree_make_string_view(source, source_length),
      .import = loom_configured_input_loomc_importer(),
  };
  iree_status_t status = loom_tooling_input_admit_loomc_module(
      &admission, npu->context, npu->workspace, &npu->block_pool,
      npu->allocator, &module, &result);
  if (iree_status_is_ok(status)) {
    status = lfm3_npu_check_result(npu, result, "admission");
  } else {
    loomc_result_release(result);
  }
  result = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_tooling_cli_prepare_loomc_pass_program(
        npu->context, module, IREE_SV("default"), IREE_SV("lfm3 npu"),
        npu->allocator, &pass_program, &result);
    if (iree_status_is_ok(status)) {
      status = lfm3_npu_check_result(npu, result, "pass program");
    } else {
      loomc_result_release(result);
    }
    result = NULL;
  }
  const loomc_string_view_t compile_root = loomc_make_cstring_view(root);
  const loomc_emit_options_t emit_options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .structure_size = sizeof(emit_options),
      .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };
  loomc_config_options_t config_options = {0};
  const loomc_compile_artifact_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS,
      .structure_size = sizeof(compile_options),
      .roots = &compile_root,
      .root_count = 1,
      .target_profile = npu->target_profile,
      .config = &config_options,
      .emit_options = &emit_options,
      .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG,
  };
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(
        loomc_compile_artifact(npu->compiler, npu->workspace, pass_program,
                               module, &compile_options, loom_allocator, &result));
  }
  const loomc_artifact_t* executable = NULL;
  const loomc_artifact_t* launch_config = NULL;
  lfm3_npu_kernel_t* kernel = NULL;
  if (iree_status_is_ok(status)) {
    for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
      const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
      if (artifact->kind == LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG) {
        launch_config = artifact;
      } else if (artifact->kind == LOOMC_ARTIFACT_KIND_EXECUTABLE) {
        executable = artifact;
      }
    }
    if (executable == NULL) {
      // Report diagnostics if compilation failed.
      loomc_result_t* failed = result;
      result = NULL;
      status = lfm3_npu_check_result(npu, failed, "compilation");
      if (iree_status_is_ok(status)) {
        status = iree_make_status(IREE_STATUS_INTERNAL,
                                  "compiler returned no executable");
      }
    }
  }
  if (iree_status_is_ok(status)) {
    kernel = (lfm3_npu_kernel_t*)calloc(1, sizeof(*kernel));
    loomc_module_function_t selected = {0};
    status = iree_status_from_loomc(loomc_module_lookup_function(
        module, loomc_make_cstring_view(root), &selected));
    if (iree_status_is_ok(status)) {
      loomc_string_view_t name = selected.symbol_name;
      loomc_module_function_export_info_t export_info = {0};
      if (loomc_module_function_try_get_export_info(module, &selected,
                                                    &export_info) &&
          (export_info.flags & LOOMC_MODULE_FUNCTION_EXPORT_FLAG_HAS_SYMBOL) &&
          !loomc_string_view_is_empty(export_info.export_symbol)) {
        name = export_info.export_symbol;
      }
      snprintf(kernel->export_name, sizeof(kernel->export_name), "%.*s",
               (int)name.size, name.data);
    }
    if (iree_status_is_ok(status)) status = lfm3_npu_launch_config(launch_config,
                                    iree_make_cstring_view(kernel->export_name),
                                    npu->allocator, kernel->workgroup_count);
  }
  if (iree_status_is_ok(status)) {
    const loom_device_artifact_t device_artifact = {
        .executable_target = npu->executable_target,
        .contents = iree_byte_sequence_from_loomc(executable->contents),
    };
    status = loom_run_hal_prepared_candidate_prepare(
        &npu->runtime, &device_artifact, npu->allocator, &kernel->candidate);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_executable_lookup_function_by_name(
        kernel->candidate.executable, iree_make_cstring_view(kernel->export_name), &kernel->function);
  }
  loomc_result_release(result);
  loomc_pass_program_release(pass_program);
  loomc_module_release(module);
  if (!iree_status_is_ok(status)) {
    free(kernel);
    return lfm3_npu_fail(npu, status);
  }
  npu->kernels[npu->kernel_count] = kernel;
  *out_kernel = npu->kernel_count++;
  return 0;
}

int lfm3_npu_alloc(lfm3_npu* npu, uint64_t bytes, int* out_buffer) {
  *out_buffer = -1;
  if (npu->buffer_count >= LFM3_NPU_MAX_BUFFERS) {
    return lfm3_npu_fail(npu, iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                               "too many buffers"));
  }
  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_pool_allocate_buffer(
      npu->runtime.device_pool,
      iree_hal_make_pool_reservation_request(bytes ? bytes : 1, 64),
      iree_infinite_timeout(), &buffer);
  if (!iree_status_is_ok(status)) return lfm3_npu_fail(npu, status);
  npu->buffers[npu->buffer_count] = buffer;
  *out_buffer = npu->buffer_count++;
  return 0;
}

int lfm3_npu_free(lfm3_npu* npu, int id) {
  if (id < 0 || id >= npu->buffer_count || !npu->buffers[id]) return 1;
  iree_hal_buffer_release(npu->buffers[id]);
  npu->buffers[id] = NULL;
  return 0;
}

// Copies between a device buffer and a mapped staging buffer on the transfer
// queue and waits.
static iree_status_t lfm3_npu_copy(lfm3_npu* npu, iree_hal_buffer_t* source,
                                   uint64_t source_offset,
                                   iree_hal_buffer_t* target,
                                   uint64_t target_offset, uint64_t bytes) {
  iree_hal_queue_t* queue = npu->runtime.transfer_queue
                                ? npu->runtime.transfer_queue
                                : npu->runtime.dispatch_queue;
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      npu->runtime.device,
      iree_hal_make_queue_family_affinity(
          iree_hal_queue_family_ordinal(iree_hal_queue_family(queue))),
      0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  uint64_t value = 1;
  const iree_hal_semaphore_list_t signal = {
      .count = 1, .semaphores = &semaphore, .payload_values = &value};
  const iree_hal_transfer_operation_t operation = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
      .copy = {.source_buffer = source,
               .source_offset = source_offset,
               .target_buffer = target,
               .target_offset = target_offset,
               .length = bytes,
               .flags = IREE_HAL_COPY_FLAG_NONE},
  };
  iree_status_t status = iree_hal_queue_transfer(
      queue, iree_hal_semaphore_list_empty(), signal, 1, &operation, NULL);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_semaphore_wait(semaphore, value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE);
  }
  iree_hal_semaphore_release(semaphore);
  return status;
}

static iree_status_t lfm3_npu_staging(lfm3_npu* npu, uint64_t bytes,
                                      iree_hal_buffer_t** out) {
  return iree_hal_pool_allocate_buffer(
      npu->runtime.staging_pool,
      iree_hal_make_pool_reservation_request(bytes ? bytes : 1, 64),
      iree_infinite_timeout(), out);
}

static iree_status_t lfm3_npu_drain(lfm3_npu* npu) {
  if (!npu->submitted) return iree_ok_status();
  return loom_run_hal_semaphore_wait(npu->timeline, npu->submitted, iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
}

int lfm3_npu_finish(lfm3_npu* npu) { return lfm3_npu_fail(npu, lfm3_npu_drain(npu)); }

int lfm3_npu_write(lfm3_npu* npu, int id, uint64_t offset, const void* data,
                   uint64_t bytes) {
  if (lfm3_npu_finish(npu)) return 1;
  if (id < 0 || id >= npu->buffer_count || !npu->buffers[id]) return 1;
  iree_hal_buffer_t* staging = NULL;
  iree_status_t status = lfm3_npu_staging(npu, bytes, &staging);
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_map_write(staging, 0, data, bytes);
  }
  if (iree_status_is_ok(status)) {
    status = lfm3_npu_copy(npu, staging, 0, npu->buffers[id], offset, bytes);
  }
  iree_hal_buffer_release(staging);
  return lfm3_npu_fail(npu, status);
}

int lfm3_npu_read(lfm3_npu* npu, int id, uint64_t offset, void* data,
                  uint64_t bytes) {
  if (lfm3_npu_finish(npu)) return 1;
  if (id < 0 || id >= npu->buffer_count || !npu->buffers[id]) return 1;
  iree_hal_buffer_t* staging = NULL;
  iree_status_t status = lfm3_npu_staging(npu, bytes, &staging);
  if (iree_status_is_ok(status)) {
    status = lfm3_npu_copy(npu, npu->buffers[id], offset, staging, 0, bytes);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_map_read(staging, 0, data, bytes);
  }
  iree_hal_buffer_release(staging);
  return lfm3_npu_fail(npu, status);
}

int lfm3_npu_enqueue(lfm3_npu* npu, int kernel_id, const int* buffers, int count) {
  if (kernel_id < 0 || kernel_id >= npu->kernel_count || count > 64) return 1;
  lfm3_npu_kernel_t* kernel = npu->kernels[kernel_id];
  iree_hal_buffer_ref_t refs[64];
  for (int i = 0; i < count; ++i) {
    if (buffers[i] < 0 || buffers[i] >= npu->buffer_count || !npu->buffers[buffers[i]]) {
      return lfm3_npu_fail(npu, iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "bad buffer %d", buffers[i]));
    }
    iree_hal_buffer_t* buffer = npu->buffers[buffers[i]];
    refs[i] = iree_hal_make_buffer_ref(buffer, 0, iree_hal_buffer_byte_length(buffer));
  }
  const iree_hal_buffer_ref_list_t bindings = {.count = (iree_host_size_t)count, .values = refs};
  uint64_t wait_value = npu->submitted, signal_value = npu->submitted + 1;
  const iree_hal_semaphore_list_t waits = {
      .count = wait_value ? 1 : 0, .semaphores = &npu->timeline, .payload_values = &wait_value};
  const iree_hal_semaphore_list_t signals = {
      .count = 1, .semaphores = &npu->timeline, .payload_values = &signal_value};
  const iree_hal_dispatch_config_t config = iree_hal_make_static_dispatch_config(
      kernel->workgroup_count[0], kernel->workgroup_count[1], kernel->workgroup_count[2]);
  iree_status_t status = iree_hal_queue_dispatch(
      npu->runtime.dispatch_queue, waits, signals, kernel->candidate.executable, kernel->function, config,
      iree_const_byte_span_empty(), bindings, NULL, IREE_HAL_DISPATCH_FLAG_NONE);
  if (iree_status_is_ok(status)) npu->submitted = signal_value;
  return lfm3_npu_fail(npu, status);
}

int lfm3_npu_dispatch(lfm3_npu* npu, int kernel_id, const int* buffers, int count) {
  if (lfm3_npu_enqueue(npu, kernel_id, buffers, count)) return 1;
  return lfm3_npu_finish(npu);
}
