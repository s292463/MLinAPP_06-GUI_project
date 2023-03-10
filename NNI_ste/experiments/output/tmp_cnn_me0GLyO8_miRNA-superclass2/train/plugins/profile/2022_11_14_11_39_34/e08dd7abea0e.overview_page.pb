�	$��X@$��X@!$��X@	�{Կ�@�{Կ�@!�{Կ�@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL$��X@'���6�?1�.�?A�M�=���?I7o��=@Y�X4���?rEagerKernelExecute 0*	}?5^�5p@2U
Iterator::Model::ParallelMapV2>�>tA}�?!u@LLWJ@)>�>tA}�?1u@LLWJ@:Preprocessing2F
Iterator::ModelhY�����?!�gFPX@Q@)��n��?1H�<�R0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��*l��?!�(7d�2@)��<��?1��}�g/@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��ȭI�?!�䟂�@)��ȭI�?1�䟂�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��B��?!8Tu��w@)ؚ����?1��"U��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�>��?!�a澞�>@)t�Lh�X�?1�44��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�7��w�?!}�I#��@)�7��w�?1}�I#��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��3���?!]�o��!@)��;��~f?1�����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 6.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�49.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t20.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9�{Կ�@Iܒ����Q@Q��&�m�6@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	'���6�?'���6�?!'���6�?      ��!       "	�.�?�.�?!�.�?*      ��!       2	�M�=���?�M�=���?!�M�=���?:	7o��=@7o��=@!7o��=@B      ��!       J	�X4���?�X4���?!�X4���?R      ��!       Z	�X4���?�X4���?!�X4���?b      ��!       JGPUY�{Կ�@b qܒ����Q@y��&�m�6@�"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilterzd?V�?!zd?V�?0"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput��ۇ���?!̟c��?0"1
model/Conv1D_4/conv1dConv2D�M�8f	�?!�r����?"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad)���Nt�?!�����?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter����g�?!�Q��?0"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits���>-W�?!�~:����?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad��z�u��?!�+"?pV�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad꽥�`A�?!u�H���?"1
model/Conv1D_2/conv1dConv2D���J|�?!����Jb�?"K
$Adam/Adam/update_8/ResourceApplyAdamResourceApplyAdam�8��?!�ov���?Q      Y@Yݘ��V+@a`�.�U@q��ܽW�:@y'�.�}f�?"�
both�Your program is MODERATELY input-bound because 6.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�49.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t20.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�26.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 