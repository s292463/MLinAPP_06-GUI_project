�	�+d�� @�+d�� @!�+d�� @	�t��Sg@�t��Sg@!�t��Sg@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�+d�� @�S:X�'�?1���9]V@AIM��f��?I��QI� @Y�{)<hv�?rEagerKernelExecute 0*	أp=
)t@2U
Iterator::Model::ParallelMapV2<L�����?!��?	�J@)<L�����?1��?	�J@:Preprocessing2F
Iterator::Model%��1 {�?!���S@)�Ŧ�B �?1�Q�?Z)7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatx��e�?!0w�:>�*@)Úʢ���?1)v�Gou&@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice2���?!+��F	@)2���?1+��F	@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateW���̖?!xk6n��@)|�Y�H��?1�w;-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipv28J^��?!)�ۭ��7@)�]J]2��?1��ʏtB@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor!�> �M|?!��;#@)!�> �M|?1��;#@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapҪ�t���?!��0���@)ګ����e?1���_�Q�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 15.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�28.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�t��Sg@IP1�U�E@Qb'�44K@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�S:X�'�?�S:X�'�?!�S:X�'�?      ��!       "	���9]V@���9]V@!���9]V@*      ��!       2	IM��f��?IM��f��?!IM��f��?:	��QI� @��QI� @!��QI� @B      ��!       J	�{)<hv�?�{)<hv�?!�{)<hv�?R      ��!       Z	�{)<hv�?�{)<hv�?!�{)<hv�?b      ��!       JGPUY�t��Sg@b qP1�U�E@yb'�44K@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��h�d��?!��h�d��?0"1
model/Conv1D_2/conv1dConv2D|U����?!���!� �?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputU�%�0�?!]�0ڎ��?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�X��m
�?!|'���M�?0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGradjG�ur�?!��U+��?"1
model/Conv1D_3/conv1dConv2D�뢕��?!8r��H�?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputߓ�@L��?!��p:�?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad�+*�n�?!��4�D�?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�8Fv8m�?!~��q��?0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad�x5���?!C�P���?Q      Y@Y      )@a     �U@q���Rv�<@y��yŋ��?"�
both�Your program is POTENTIALLY input-bound because 15.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�28.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�28.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 