#include <vector>

#include "caffe/custom_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossWithIndexLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";

  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanLossWithIndexLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    const Dtype* label = bottom[0]->cpu_data();
    const Dtype* bottom_ideal = bottom[1]->cpu_data();
    const Dtype* bottom_estimate = bottom[2]->cpu_data();

    Dtype* top_data = top[0]->mutable_cpu_data();
    Dtype* diff1 = diff_.mutable_cpu_data();

    int num = bottom[0]->count();

    for (int i = 0; i < num; ++i) {
          diff1[i] = bottom_ideal[i] -  bottom_estimate[i*num + (int)label[i]];
          top_data[i] += diff1[i]*diff1[i] / num / Dtype(2);
    }
}

template <typename Dtype>
void EuclideanLossWithIndexLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    const Dtype* label = bottom[0]->cpu_data();
    Dtype* bottom_estimate_diff = bottom[2]->mutable_cpu_diff();
    const Dtype* diff1 = diff_.cpu_data();
   int num = bottom[0]->count();

      for (int i = 0; i < num; ++i) {
          const Dtype sign = (i == 0) ? 1 : -1;
          const Dtype alpha = sign * top[2]->cpu_diff()[0] / bottom[i]->num();
          bottom_estimate_diff[i*num + (int)label[i]] = alpha *diff1[i];
      }
}

//#ifdef CPU_ONLY
//STUB_GPU(EuclideanLossWithIndexLayer);
//#endif

INSTANTIATE_CLASS(EuclideanLossWithIndexLayer);
REGISTER_LAYER_CLASS(EuclideanLossWithIndex);

}  // namespace caffe

