�	��̒ �d@��̒ �d@!��̒ �d@      ��!       "{
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
�9�6@y5H��1]S@�"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput0�����?!0�����?0"1
model/Conv1D_3/conv1dConv2DD�G��`�?!A��$��?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilters�����?!/�c5�?0"1
model/Conv1D_4/conv1dConv2D�R��Ua�?!������?"1
model/Conv1D_2/conv1dConv2D�J�UW=�?!�ay9)�?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterHM!b��?!�����=�?0"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter!���\�?!����e�?0"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput,0�qB�?!\��	���?0"1
model/Conv1D_1/conv1dConv2D3�Xuڪ?!��t_4��?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputc�MQ��?!%�Pt1�?0Q      Y@YAva1J@a����^�W@q�^�j�B@y��#~W^k?"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�22.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�37.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 