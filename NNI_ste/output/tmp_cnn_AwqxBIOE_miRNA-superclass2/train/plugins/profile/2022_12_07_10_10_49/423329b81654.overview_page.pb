�	���,�-@���,�-@!���,�-@	��帡�@��帡�@!��帡�@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC���,�-@i�x�Jx�?1'��rJ�'@I&���^�?YJ�?����?rEagerKernelExecute 0*	����Mjd@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat0��s�?!b��l�]A@)Q���Y�?1�-�(P>@:Preprocessing2F
Iterator::Model��ek}�?!'��m�D@)���9"ߥ?1� �u�':@:Preprocessing2U
Iterator::Model::ParallelMapV2�H�"i7�?!2K< Z/@)�H�"i7�?12K< Z/@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceG�P�[�?!V�K
�%'@)G�P�[�?1V�K
�%'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��E}�;�?!l�,ҡ�0@)j��{��?1�4O;@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��R�?!��`B�M@)��8h�?1��J��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�����}?!D�u�Ϯ@)�����}?1D�u�Ϯ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapU��C��?!�X��2@)�"�~�f?1�w�^X_�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 6.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��帡�@I|�	e*(1@Q,U�X��S@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	i�x�Jx�?i�x�Jx�?!i�x�Jx�?      ��!       "	'��rJ�'@'��rJ�'@!'��rJ�'@*      ��!       2      ��!       :	&���^�?&���^�?!&���^�?B      ��!       J	J�?����?J�?����?!J�?����?R      ��!       Z	J�?����?J�?����?!J�?����?b      ��!       JGPUY��帡�@b q|�	e*(1@y,U�X��S@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter@����7�?!@����7�?0"1
model/Conv1D_2/conv1dConv2DRmK�6��?!ɯ�H
p�?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�I�k�&�?!�Tl���?0"1
model/Conv1D_3/conv1dConv2D��[��?!	���}��?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��2~�?!�5.Dt�?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput_	�V��?!3���nK�?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput"R���O�?!wr��g��?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradP "�ƣ?!s�8N�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad�ʿ�5�?!Wl����?"3
model/Conv1D_1/BiasAddBiasAdd��fq��?!��z�D�?Q      Y@Y&W�+�)@a�����U@q-�1��!@y��E�+آ?"�

both�Your program is POTENTIALLY input-bound because 6.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 