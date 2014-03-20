// Copyright 2014 kloudkl@github

#include <stdint.h>  // for uint32_t & uint64_t

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template<typename Dtype>
class MathFunctionsTest : public ::testing::Test {
 protected:
  MathFunctionsTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
  }

  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    this->blob_bottom_->Reshape(100, 70, 50, 30);
    this->blob_top_->Reshape(100, 70, 50, 30);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_top_);
  }

  virtual ~MathFunctionsTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  // http://en.wikipedia.org/wiki/Hamming_distance
  int ReferenceHammingDistance(const int n, const Dtype* x, const Dtype* y);

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
};

#define REF_HAMMING_DIST(float_type, int_type) \
template<> \
int MathFunctionsTest<float_type>::ReferenceHammingDistance(const int n, \
                                                       const float_type* x, \
                                                       const float_type* y) { \
  int dist = 0; \
  int_type val; \
  for (int i = 0; i < n; ++i) { \
    val = static_cast<int_type>(x[i]) ^ static_cast<int_type>(y[i]); \
    /* Count the number of set bits */ \
    while (val) { \
      ++dist; \
      val &= val - 1; \
    } \
  } \
  return dist; \
}

REF_HAMMING_DIST(float, uint32_t);
REF_HAMMING_DIST(double, uint64_t);

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(MathFunctionsTest, Dtypes);

TYPED_TEST(MathFunctionsTest, TestHammingDistance) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  const TypeParam* y = this->blob_top_->cpu_data();
  CHECK_EQ(this->ReferenceHammingDistance(n, x, y),
           caffe_hamming_distance<TypeParam>(n, x, y));
}

}  // namespace caffe