#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/custom_layers.hpp"

namespace caffe {

template <typename Dtype>
void ELULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * (exp(bottom_data[i]) - Dtype(1));
  }
}

template <typename Dtype>
void ELULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* top_data = top[0]->cpu_data();
  
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  const Dtype* top_diff = top[0]->cpu_diff();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

   // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)  + (bottom_data[i] <= 0) * (top_data[i] + negative_slope));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ELULayer);
#endif

INSTANTIATE_CLASS(ELULayer);
REGISTER_LAYER_CLASS(ELU);

}  // namespace caffe