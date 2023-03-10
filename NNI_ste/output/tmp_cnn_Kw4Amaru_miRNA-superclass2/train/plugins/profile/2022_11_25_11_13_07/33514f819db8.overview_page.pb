�	�f��js(@�f��js(@!�f��js(@	B�]�{@B�]�{@!B�]�{@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�f��js(@�}(F��?1*���P$@A�St$��p?I�|���]�?Y�U�0�{�?rEagerKernelExecute 0*	H�z��j@2F
Iterator::Model*����1�?!����G@)� ��z�?1T��#��>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat.�5#��?!C��j�9@)S�G�?18�`��5@:Preprocessing2U
Iterator::Model::ParallelMapV2~!<�8�?!�e5R(0@)~!<�8�?1�e5R(0@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����G�?!���Ot!@)����G�?1���Ot!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��׺��?!�pEIJ@)4��<��?1���9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�3�ތ��?!R7W�}�,@)�EB[Υ�?1kx��[P@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapf2�g@�?!�n@�F=3@)�a0�̅?1�LS. �@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor̖��p��?!U fR��@)̖��p��?1U fR��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 5.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9C�]�{@I��ώNY,@Q��QڋT@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�}(F��?�}(F��?!�}(F��?      ��!       "	*���P$@*���P$@!*���P$@*      ��!       2	�St$��p?�St$��p?!�St$��p?:	�|���]�?�|���]�?!�|���]�?B      ��!       J	�U�0�{�?�U�0�{�?!�U�0�{�?R      ��!       Z	�U�0�{�?�U�0�{�?!�U�0�{�?b      ��!       JGPUYC�]�{@b q��ώNY,@y��QڋT@�"1
model/Conv1D_3/conv1dConv2D�Jl�M�?!�Jl�M�?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�8��?!k��j��?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter����]��?!�C�����?0"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFiltervxÏr��?!�L'��?0"1
model/Conv1D_2/conv1dConv2D�;(�¸�?!F���?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput&�m���?!�֥0%��?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputɚȌj�?!�/��v�?0"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput������?!j�f�
�?0"1
model/Conv1D_4/conv1dConv2D�唶��?!�=��Ϛ�?"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad����lc�?!@G=��?Q      Y@Y?]��O�'@aX,
V@q jؗO�"@y��C�$�?"�

both�Your program is POTENTIALLY input-bound because 5.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�8.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 