�	�H�5�@�H�5�@!�H�5�@	a�X��S@a�X��S@!a�X��S@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�H�5�@	^��?1�8����?A3���U֖?I)����	@Ywi�ai��?rEagerKernelExecute 0*	���K�h@2F
Iterator::Model�B:<��?!� ��C@)�)H�?1ސ8�<@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate*��g\�?!�.��D>@@)</O犪?1n��DZ:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatɪ7U�?!g��v�+6@)OqN`�?1�Zf�t>2@:Preprocessing2U
Iterator::Model::ParallelMapV2�1˞6�?!UfƒC'@)�1˞6�?1UfƒC'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceB]¡��?!"%�k�@)B]¡��?1"%�k�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip����?k�?!�@�73N@)�4�Op�?11��0P@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�3��X�?!�Mcsl@)�3��X�?1�Mcsl@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap\�O��?!� �6�@@)�
�.�f?1~�\G7��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 23.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�45.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9a�X��S@Ii6KQ@Q�=��;@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
		^��?	^��?!	^��?      ��!       "	�8����?�8����?!�8����?*      ��!       2	3���U֖?3���U֖?!3���U֖?:	)����	@)����	@!)����	@B      ��!       J	wi�ai��?wi�ai��?!wi�ai��?R      ��!       Z	wi�ai��?wi�ai��?!wi�ai��?b      ��!       JGPUYa�X��S@b qi6KQ@y�=��;@�"1
model/Conv1D_2/conv1dConv2D��2�F�?!��2�F�?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter������?!�P����?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputiM!�ܩ?!7�Y���?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradp��+�?!��z��Q�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradz�韚�?!�Y��>��?"1
model/Conv1D_1/conv1dConv2DOQ>tz�?!�#<���?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��q8�ޟ?!�>�̘��?0"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	TransposeQ�hr/f�?!'������?"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transposel�1j40�?!��
�M�?"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput���¨�?!��]3��?0Q      Y@Ym۶m۶)@a�$I�$�U@q��CdW�9@yQV�O'$�?"�
both�Your program is POTENTIALLY input-bound because 23.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�45.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�25.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 