�	��s�� @��s�� @!��s�� @      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC��s�� @����A��?1X˝�`x@A�A�F��?I�\QJ�	@rEagerKernelExecute 0*	�A`��g@2F
Iterator::Model9�)9'��?!W�R��1F@)ȳ˷>�?1���5�=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatM1AG��?!�.��<<@)F�Swe�?1l� ��8@:Preprocessing2U
Iterator::Model::ParallelMapV2�$@M-[�?!�箝�,@)�$@M-[�?1�箝�,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����S��?!�r� ��&@)����S��?1�r� ��&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�t?� ?�?!O���`4@)�Y���А?1֛_!�!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��(_�B�?!�Y�=:�K@)�MbX9�?1l߾��i@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor6�Ko.z?!���f��@)6�Ko.z?1���f��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapFD1y̤?!��v�'6@)��-�h?1�U�(�C�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 15.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�38.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI����3K@Q'��F@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����A��?����A��?!����A��?      ��!       "	X˝�`x@X˝�`x@!X˝�`x@*      ��!       2	�A�F��?�A�F��?!�A�F��?:	�\QJ�	@�\QJ�	@!�\QJ�	@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q����3K@y'��F@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter
�hd$�?!
�hd$�?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput��>Jjl�?!P�SWǦ�?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter����Q�?!���7��?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput"d�8l�?!��aF�?0"1
model/Conv1D_2/conv1dConv2D9��лU�?!��ƅڵ�?"1
model/Conv1D_3/conv1dConv2D૪S�m�?!�����?"C
%gradient_tape/model/Conv1D_2/ReluGradReluGrad��@���?!���r]�?"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad������?!i�}�A��?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad}.W3W�?!Q��#�n�?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad���iQZ�?!\�:Z��?Q      Y@Y&W�+�)@a�����U@qw�*��;@y��qq-�?"�
both�Your program is POTENTIALLY input-bound because 15.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�38.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�27.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 