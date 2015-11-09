#include <vector>

#include "caffe/custom_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossWithIndexLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "Index and Target inputs must have the same dimension.";

  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanLossWithIndexLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    const Dtype* index = bottom[0]->cpu_data();
    const Dtype* bottom_ideal = bottom[1]->cpu_data();
    const Dtype* bottom_estimate = bottom[2]->cpu_data();

    Dtype* top_data = top[0]->mutable_cpu_data();
    Dtype* diff1 = diff_.mutable_cpu_data();

    int num = bottom[0]->count();
    caffe_set(diff_.count(), Dtype(0), diff1);
    top_data[0] = 0;
    for (int i = 0; i < num; ++i) {
          diff1[i] = bottom_ideal[i] -  bottom_estimate[i*num + (int)index[i]];
          top_data[0] += diff1[i] * diff1[i] / Dtype(num) / Dtype(2);
    }
}

template <typename Dtype>
void EuclideanLossWithIndexLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to index inputs.";
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[2]) {
    const Dtype* index = bottom[0]->cpu_data();
    Dtype* bottom_estimate_diff = bottom[2]->mutable_cpu_diff();
    const Dtype* diff1 = diff_.cpu_data();
    int num = bottom[0]->count();
    caffe_set(bottom[2]->count(), Dtype(0), bottom_estimate_diff);
    const Dtype alpha = top[0]->cpu_diff()[0] / num;
      for (int i = 0; i < num; ++i) {
          bottom_estimate_diff[i*num + (int)index[i]] = alpha *diff1[i];
      }
  }
}

//#ifdef CPU_ONLY
//STUB_GPU(EuclideanLossWithIndexLayer);
//#endif

INSTANTIATE_CLASS(EuclideanLossWithIndexLayer);
REGISTER_LAYER_CLASS(EuclideanLossWithIndex);

}  // namespace caffe

