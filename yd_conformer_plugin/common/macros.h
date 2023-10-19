/*************************************************************************
 *  * Copyright (C) [2020-2023] by Cambricon, Inc.
 *   *************************************************************************/
#ifndef SAMPLES_CC_BASIC_SAMPLES_SAMPLE_PLUGINOP_COMMON_MACROS_H_
#define SAMPLES_CC_BASIC_SAMPLES_SAMPLE_PLUGINOP_COMMON_MACROS_H_
#include <string>
#include "mm_runtime.h"  // NOLINT

#define PLUGIN_MM_CHECK(pluginop, status) \
  do {                                    \
    auto ret = (status);                  \
    if (ret != magicmind::Status::OK()) { \
      return ret;                         \
    }                                     \
  } while (0)

#define PLUGIN_CNNL_CHECK(pluginop, cnnl_api_call)                                                \
  do {                                                                                            \
    cnnlStatus_t ret = (cnnl_api_call);                                                           \
    if (ret != CNNL_STATUS_SUCCESS) {                                                             \
      std::string temp =                                                                          \
          std::string(pluginop) + "[" + std::string(__FUNCTION__) + "()] : CNNL api call failed"; \
      magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);                        \
      return status;                                                                              \
    }                                                                                             \
  } while (0)

  

#endif  // SAMPLES_CC_BASIC_SAMPLES_SAMPLE_PLUGINOP_COMMON_MACROS_H_

