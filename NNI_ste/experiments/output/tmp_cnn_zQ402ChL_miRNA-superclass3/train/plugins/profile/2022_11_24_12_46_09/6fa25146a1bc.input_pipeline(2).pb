	��̒ �d@��̒ �d@!��̒ �d@      ��!       "{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:��̒ �d@��cw��?1u9% &`@Iw;S�B@rEagerKernelExecute 0*	�x�&1�c@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�	0,��?!����iB@)���r��?1��i�\A@:Preprocessing2F
Iterator::Model�6U���?!�`.y�fG@)�'��?1xg���=@:Preprocessing2U
Iterator::Model::ParallelMapV2a�ri�?!NZ�cv/1@)a�ri�?1NZ�cv/1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeata7l[�ِ?!J�"��$@)zm6Vb��?1 a=��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�7j��{�?!�цB�J@)���x}?1B}qM>@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�D��)x?!�X� �@)�D��)x?1�X� �@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapQ����ۮ?!#G��XC@)�y�ؘ�a?1m-����?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor�bc^G\?!���ki��?)�bc^G\?1���ki��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice����Y?!�I><Q�?)����Y?1�I><Q�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�22.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI,�
�9�6@Q5H��1]S@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��cw��?��cw��?!��cw��?      ��!       "	u9% &`@u9% &`@!u9% &`@*      ��!       2      ��!       :	w;S�B@w;S�B@!w;S�B@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q,�
�9�6@y5H��1]S@