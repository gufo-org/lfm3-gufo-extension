# LFM3 1.7B, LFM3-VL 2B, LFM3-Audio 2B and d1-Omni 2B external model extension.

gufo_extension_target(
  NAME gufo_lfm3
  SOURCES
    config.cpp
    continuation_layout.cpp
    weights.cpp
    tokenizer.cpp
    chat_template.cpp
    reference.cpp
    probe.cpp
    extension.cpp
    vl/image.cpp
    vl/prompt.cpp
    d1/prompt.cpp
    d1/packing.cpp
    d1/boxes.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    ICU::uc
    ICU::i18n
    Threads::Threads
)

if(TARGET gufo_lfm3)
  target_include_directories(gufo_lfm3 PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
  if(ENGINE_ENABLE_HIP)
    target_compile_definitions(gufo_lfm3 PRIVATE ENGINE_ENABLE_HIP=1)
  endif()
endif()

gufo_extension_target(
  NAME gufo_lfm3_hip
  SOURCES
    engine.cpp
    upload.cpp
    npu/engine.cpp
    npu/runtime.cpp
  LINK_LIBRARIES
    gufo_lfm3
    gufo_ext_runtime
    hip::host
    roc::hipblas
    roc::hipblaslt
    Threads::Threads
)

if(TARGET gufo_lfm3_hip)
  gufo_extension_hip_sources(gufo_lfm3_hip
    SOURCES
      hip/ops.hip
  )
  target_compile_options(gufo_lfm3_hip PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-command-line-argument>
  )
  target_include_directories(gufo_lfm3_hip PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
endif()

# LFM3-Audio towers and serving runners over the HIP backbone.
gufo_extension_target(
  NAME gufo_lfm3_audio
  SOURCES
    audio/weights.cpp
    audio/audio.cpp
    audio/model.cpp
    audio/generation.cpp
    audio/realtime.cpp
    audio/runner.cpp
    audio/speak.cpp
    audio/provider.cpp
  LINK_LIBRARIES
    roc::hipblas
    gufo_lfm3_hip
    gufo_lfm3
    gufo_ext_runtime
    hip::host
    Threads::Threads
)

if(TARGET gufo_lfm3_audio)
  gufo_extension_hip_sources(gufo_lfm3_audio
    SOURCES
      audio/hip/ops.hip
  )
  target_compile_options(gufo_lfm3_audio PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-command-line-argument>
  )
endif()

# LFM3-VL vision tower; shares the audio encoder's operators.
gufo_extension_target(
  NAME gufo_lfm3_vl
  SOURCES
    vl/vision.cpp
  LINK_LIBRARIES
    gufo_lfm3_audio
    gufo_lfm3_hip
    gufo_lfm3
    gufo_ext_runtime
    hip::host
)

if(TARGET gufo_lfm3_vl)
  gufo_extension_hip_sources(gufo_lfm3_vl
    SOURCES
      vl/hip/ops.hip
  )
  target_compile_options(gufo_lfm3_vl PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-command-line-argument>
  )
endif()

# Text and image generation runner, CLI drivers and the text provider.
gufo_extension_target(
  NAME gufo_lfm3_text
  SOURCES
    continuation_adapter.cpp
    runner.cpp
    drivers.cpp
    provider.cpp
  LINK_LIBRARIES
    gufo_lfm3_vl
    gufo_lfm3_hip
    gufo_lfm3
    gufo_ext_cache
    gufo_ext_runtime
    hip::host
    Threads::Threads
)

if(TARGET gufo_lfm3_text)
  target_compile_options(gufo_lfm3_text PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
endif()

# d1-Omni decisions over text, the LFM3-VL vision tower and the audio
# encoder.
gufo_extension_target(
  NAME gufo_lfm3_d1
  SOURCES
    d1/model.cpp
    d1/runner.cpp
    d1/provider.cpp
  LINK_LIBRARIES
    gufo_lfm3_vl
    gufo_lfm3_audio
    gufo_lfm3_hip
    gufo_lfm3
    gufo_ext_runtime
    hip::host
)

if(TARGET gufo_lfm3_d1)
  target_compile_options(gufo_lfm3_d1 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
endif()

# gufo_ext_runtime only declares the public services; these link their
# implementations into executables that stand apart from `gufo`.
set(_gufo_lfm3_test_libs gufo_ext_core gufo_ext_cache)
if(ENGINE_ENABLE_HIP)
  list(APPEND _gufo_lfm3_test_libs gufo_ext_hip)
endif()

gufo_extension_target(
  NAME lfm3_config_test
  SOURCES
    tests/config_test.cpp
  LINK_LIBRARIES
    gufo_lfm3
    gufo_ext_runtime
    ${_gufo_lfm3_test_libs}
)

if(TARGET lfm3_config_test)
  add_test(NAME lfm3.config COMMAND lfm3_config_test)
  set_tests_properties(lfm3.config PROPERTIES LABELS "lfm3;cpu")
endif()

gufo_extension_target(
  NAME lfm3_ops_test
  SOURCES
    tests/ops_test.cpp
  LINK_LIBRARIES
    gufo_lfm3_hip
    gufo_lfm3
    gufo_ext_runtime
    ${_gufo_lfm3_test_libs}
    hip::host
)

if(TARGET lfm3_ops_test)
  add_test(NAME lfm3.ops COMMAND lfm3_ops_test)
  set_tests_properties(lfm3.ops PROPERTIES LABELS "lfm3;hip;gpu;gfx1151")
endif()

gufo_extension_target(
  NAME lfm3_continuation_layout_test
  SOURCES
    tests/continuation_layout_test.cpp
  LINK_LIBRARIES
    gufo_lfm3
    gufo_ext_runtime
    ${_gufo_lfm3_test_libs}
)

gufo_extension_target(
  NAME lfm3_continuation_adapter_test
  SOURCES
    tests/continuation_adapter_test.cpp
  LINK_LIBRARIES
    gufo_lfm3_text
    gufo_lfm3_hip
    gufo_lfm3
    gufo_ext_runtime
    ${_gufo_lfm3_test_libs}
    hip::host
)

gufo_extension_target(
  NAME lfm3_session_test
  SOURCES
    tests/session_test.cpp
  LINK_LIBRARIES
    gufo_lfm3_hip
    gufo_lfm3
    gufo_ext_runtime
    ${_gufo_lfm3_test_libs}
    hip::host
)

if(ENGINE_ENABLE_HIP)
  set(_gufo_lfm3_probe_libs gufo_lfm3_hip gufo_lfm3 gufo_ext_runtime
    ${_gufo_lfm3_test_libs} hip::host)
else()
  set(_gufo_lfm3_probe_libs gufo_lfm3 gufo_ext_runtime ${_gufo_lfm3_test_libs})
endif()

gufo_extension_target(
  NAME lfm3_probe
  SOURCES
    tests/probe.cpp
  LINK_LIBRARIES
    ${_gufo_lfm3_probe_libs}
)

gufo_extension_target(
  NAME lfm3_prefix_probe
  SOURCES
    tests/prefix_probe.cpp
  LINK_LIBRARIES
    gufo_lfm3_hip
    gufo_lfm3
    gufo_ext_runtime
    ${_gufo_lfm3_test_libs}
    hip::host
)

# LFM3-Audio host helpers: WAV, resampling, prompt framing; no weights.
gufo_extension_target(
  NAME lfm3_audio_test
  SOURCES
    tests/audio_test.cpp
  LINK_LIBRARIES
    gufo_lfm3_audio
    gufo_lfm3_hip
    gufo_lfm3
    gufo_ext_runtime
    ${_gufo_lfm3_test_libs}
    hip::host
)

if(TARGET lfm3_audio_test)
  add_test(NAME lfm3_audio.host COMMAND lfm3_audio_test)
  set_tests_properties(lfm3_audio.host PROPERTIES LABELS "lfm3-audio;hip")
endif()

# Needs weights and reference fixtures; see docs/audio/QUALITY.md.
gufo_extension_target(
  NAME lfm3_audio_probe
  SOURCES
    tests/audio_probe.cpp
  LINK_LIBRARIES
    gufo_lfm3_audio
    gufo_lfm3_hip
    gufo_lfm3
    gufo_ext_runtime
    ${_gufo_lfm3_test_libs}
    hip::host
)

# Transcription latency and the shared prompt pass; needs weights.
gufo_extension_target(
  NAME lfm3_asr_probe
  SOURCES
    tests/asr_probe.cpp
  LINK_LIBRARIES
    gufo_lfm3_audio
    gufo_lfm3_hip
    gufo_lfm3
    gufo_ext_runtime
    ${_gufo_lfm3_test_libs}
    hip::host
)

gufo_extension_target(
  NAME lfm3_audio_batch_bench
  SOURCES
    tests/audio_batch_bench.cpp
  LINK_LIBRARIES
    gufo_lfm3_audio
    gufo_lfm3_hip
    gufo_lfm3
    gufo_ext_runtime
    ${_gufo_lfm3_test_libs}
    hip::host
)

# LFM3-VL template placement and image preprocessing; no weights.
gufo_extension_target(
  NAME lfm3_vl_contract_test
  SOURCES
    tests/vl_contract_test.cpp
  LINK_LIBRARIES
    gufo_lfm3
    gufo_ext_runtime
    ${_gufo_lfm3_test_libs}
)

if(TARGET lfm3_vl_contract_test)
  add_test(NAME lfm3_vl.contract COMMAND lfm3_vl_contract_test)
  set_tests_properties(lfm3_vl.contract PROPERTIES LABELS "lfm3-vl;cpu")
endif()

# Writes native prompt IDs, image embeddings and logits for the reference;
# see tests/vl_reference/README.md.
gufo_extension_target(
  NAME lfm3_vl_probe
  SOURCES
    tests/vl_probe.cpp
  LINK_LIBRARIES
    gufo_lfm3_vl
    gufo_lfm3_audio
    gufo_lfm3_hip
    gufo_lfm3
    gufo_ext_runtime
    ${_gufo_lfm3_test_libs}
    hip::host
)

# d1 prompt layout, packing, box readout and calibration; no weights.
gufo_extension_target(
  NAME d1_contract_test
  SOURCES
    tests/d1_contract_test.cpp
  LINK_LIBRARIES
    gufo_lfm3
    gufo_ext_runtime
    ${_gufo_lfm3_test_libs}
)

if(TARGET d1_contract_test)
  add_test(NAME d1.contract COMMAND d1_contract_test)
  set_tests_properties(d1.contract PROPERTIES LABELS "d1;cpu")
endif()

gufo_extension_target(
  NAME d1_ops_test
  SOURCES
    tests/d1_ops_test.cpp
  LINK_LIBRARIES
    gufo_lfm3_vl
    gufo_lfm3_audio
    gufo_lfm3_hip
    gufo_lfm3
    gufo_ext_runtime
    ${_gufo_lfm3_test_libs}
    hip::host
)

if(TARGET d1_ops_test)
  add_test(NAME d1.ops COMMAND d1_ops_test)
  set_tests_properties(d1.ops PROPERTIES LABELS "d1;hip;gpu;gfx1151")
endif()

# Writes native logits for the reference; see tests/d1_reference/README.md.
gufo_extension_target(
  NAME d1_forward_probe
  SOURCES
    tests/d1_forward_probe.cpp
  LINK_LIBRARIES
    gufo_lfm3_hip
    gufo_lfm3
    gufo_ext_runtime
    ${_gufo_lfm3_test_libs}
    hip::host
)
