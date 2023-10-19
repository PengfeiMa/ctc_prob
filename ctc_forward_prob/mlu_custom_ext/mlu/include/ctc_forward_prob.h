#ifndef CAMBRICON_CTC_FORWARD_PROB_H
#define CAMBRICON_CTC_FORWARD_PROB_H
#include <cnrt.h>

template <typename T>
void ctc_forward_prob_kernel_entry(cnrtQueue_t queue, T *r, T *s, T *x, T *output,
                                   size_t start, size_t end, size_t num_per_compute);

#endif // CAMBRICON_CTC_FORWARD_PROB_H