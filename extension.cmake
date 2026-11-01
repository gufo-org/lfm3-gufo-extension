# LFM3 1.7B external model extension.

gufo_extension_target(
  NAME gufo_lfm3
  SOURCES
    config.cpp
    weights.cpp
    tokenizer.cpp
    chat_template.cpp
    reference.cpp
    probe.cpp
    extension.cpp
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
    runner.cpp
    drivers.cpp
    provider.cpp
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
    $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
    $<$<COMPILE_LANGUAGE:HIP>:-fno-finite-math-only>
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-command-line-argument>
  )
  target_include_directories(gufo_lfm3_hip PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
endif()

gufo_extension_target(
  NAME lfm3_config_test
  SOURCES
    tests/config_test.cpp
  LINK_LIBRARIES
    gufo_lfm3
    gufo_ext_runtime
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
    hip::host
)

if(TARGET lfm3_ops_test)
  add_test(NAME lfm3.ops COMMAND lfm3_ops_test)
  set_tests_properties(lfm3.ops PROPERTIES LABELS "lfm3;hip;gpu;gfx1151")
endif()

gufo_extension_target(
  NAME lfm3_session_test
  SOURCES
    tests/session_test.cpp
  LINK_LIBRARIES
    gufo_lfm3_hip
    gufo_lfm3
    gufo_ext_runtime
    hip::host
)

if(ENGINE_ENABLE_HIP)
  set(_gufo_lfm3_probe_libs gufo_lfm3_hip gufo_lfm3 gufo_ext_runtime hip::host)
else()
  set(_gufo_lfm3_probe_libs gufo_lfm3 gufo_ext_runtime)
endif()

gufo_extension_target(
  NAME lfm3_probe
  SOURCES
    tests/probe.cpp
  LINK_LIBRARIES
    ${_gufo_lfm3_probe_libs}
)

if(ENGINE_ENABLE_HIP AND TARGET lfm3_probe)
  target_compile_definitions(lfm3_probe PRIVATE ENGINE_ENABLE_HIP=1)
endif()
