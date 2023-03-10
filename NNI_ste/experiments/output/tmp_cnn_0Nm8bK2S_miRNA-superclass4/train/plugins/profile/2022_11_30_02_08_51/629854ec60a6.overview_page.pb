�	�lV}�r2@�lV}�r2@!�lV}�r2@	��T�u�?��T�u�?!��T�u�?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC�lV}�r2@1#�=��?1��2p�.@I����Ǧ�?Yq���h �?rEagerKernelExecute 0*	+�e@2F
Iterator::Model�}U.T��?!Klë�|I@)���jdW�?1�G��چ>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��Y����?!)���>@)�2Wզ?1���Ͼu:@:Preprocessing2U
Iterator::Model::ParallelMapV2X��C��?!Đ�ks4@)X��C��?1Đ�ks4@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�����y�?!~�*M�4@)�����y�?1~�*M�4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�����?!��s��z(@)L�$zł?1�#���@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��#�&�?!��<T�H@)��rf�B?1��Dn%@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�_>Y1\}?!\��(9@)�_>Y1\}?1\��(9@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapx�g�ɗ?!D��e��+@)��_�Le?1H�L\���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 4.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��T�u�?I ��%��-@Q��97�T@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	1#�=��?1#�=��?!1#�=��?      ��!       "	��2p�.@��2p�.@!��2p�.@*      ��!       2      ��!       :	����Ǧ�?����Ǧ�?!����Ǧ�?B      ��!       J	q���h �?q���h �?!q���h �?R      ��!       Z	q���h �?q���h �?!q���h �?b      ��!       JGPUY��T�u�?b q ��%��-@y��97�T@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�T�K�>�?!�T�K�>�?0"1
model/Conv1D_2/conv1dConv2D�����K�?!0k(�sE�?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput����?!,!zo��?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter/ jdy0�?!֝��j�?0"1
model/Conv1D_3/conv1dConv2D����!��?!�)����?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad�*�.i�?!�����?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad�E�puܠ?!�������?"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose�/f���?!��+���?"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transposel���l�?!�;�5Zt�?"3
model/Conv1D_1/BiasAddBiasAdd	�u��?!|�,V�?Q      Y@Y@n]�G*@a8R4��U@q����*	*@y���?ً�?"�
both�Your program is POTENTIALLY input-bound because 4.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�13.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 