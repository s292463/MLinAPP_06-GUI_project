�	�wD @�wD @!�wD @	'k\���?'k\���?!'k\���?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�wD @DkE����?1���>9��?A b���4�?I�/����@Y���z0�?rEagerKernelExecute 0*	��C�l�b@2F
Iterator::Model�;k�]h�?!����!H@)/��d�۪?13b� �A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatp&����?!�
g�B1:@)H1@�	�?1�����c6@:Preprocessing2U
Iterator::Model::ParallelMapV2�p��?!�b5D*@)�p��?1�b5D*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip<FzQ��?!Y.�I@)�� 4J��?1�P��U$@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�J�4�?!�=����@)�J�4�?1�=����@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�j���?!�C`���+@)���y7�?1�I(�XU@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;�/K;5w?!�\x��l@);�/K;5w?1�\x��l@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapȳ˷>��?!Xʙ��/@)j0�G�d?1�5�9�9�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 23.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�55.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9'k\���?I��]��S@Q���~�3@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	DkE����?DkE����?!DkE����?      ��!       "	���>9��?���>9��?!���>9��?*      ��!       2	 b���4�? b���4�?! b���4�?:	�/����@�/����@!�/����@B      ��!       J	���z0�?���z0�?!���z0�?R      ��!       Z	���z0�?���z0�?!���z0�?b      ��!       JGPUY'k\���?b q��]��S@y���~�3@�"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput���ɽ�?!���ɽ�?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterΟsՄ]�?!��N���?0"1
model/Conv1D_2/conv1dConv2DhR߉���?!;�<�
_�?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterÔ��?!��8 \�?0"1
model/Conv1D_3/conv1dConv2D�ޘ��?!�'06���?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputLx:��?!���y�-�?0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad0Z��/�?! ��n�-�?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad��VJ�ҟ?!>�-C�*�?"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad<8Sn؛?!rqa(s��?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradA i2tj�?!v�k��?Q      Y@Y&W�+�)@a�����U@q�_�M�E@y���oO�?"�
both�Your program is POTENTIALLY input-bound because 23.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�55.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�44.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 