� 	+�~N9 @+�~N9 @!+�~N9 @	�@�4��@�@�4��@!�@�4��@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL+�~N9 @T����?1Ug��@A��u6䟙?I����@Y X9��?rEagerKernelExecute 0*	0�$���@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap����O��?!�./��D@)"U����?1��b��B@:Preprocessing2U
Iterator::Model::ParallelMapV2�+���?!�)\� �6@)�+���?1�)\� �6@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map�PۆQ�?!�(����8@)פ����?1�;H\��1@:Preprocessing2F
Iterator::Model�(�A&�?!15�Z�=@)�W�\�?1�d���@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatu���a��?!���ub@)�����	�?1�7�]o�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateIG9�M��?!���@� @)��:r�3�?1D
U����?:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat<0��?! �=��@)����L0�?1�������?:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch#gaO;��?!�Q�����?)#gaO;��?1�Q�����?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��"�-��?!�R׏�F@)��Iط�?1`C3���?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���b('z?!K>��G��?)���b('z?1K>��G��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice%��ID�w?!a�s���?)%��ID�w?1a�s���?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeU�����u?!p��?)U�����u?1p��?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateOw�x��?!�B�����?)��N�j`?1&��u�G�?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor�t�_��T?!�|�����?)�t�_��T?1�|�����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 17.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�38.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�@�4��@I4P�8��K@Q���#9�D@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	T����?T����?!T����?      ��!       "	Ug��@Ug��@!Ug��@*      ��!       2	��u6䟙?��u6䟙?!��u6䟙?:	����@����@!����@B      ��!       J	 X9��? X9��?! X9��?R      ��!       Z	 X9��? X9��?! X9��?b      ��!       JGPUY�@�4��@b q4P�8��K@y���#9�D@�"1
model/Conv1D_2/conv1dConv2Dp�,�C�?!p�,�C�?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterK�6��?!^屁�q�?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter~�X��C�?!�#�뷂�?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradv�x!��?!�W�n��?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput'����k�?!�C.�5��?0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradBR�1�?!ۆ�O�?"1
model/Conv1D_3/conv1dConv2D��7���?!l�?vY��?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputs$0o���?!��%�E�?0"1
model/Conv1D_1/conv1dConv2D*Œ���?!��w���?"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose.�B���?!�7 �
��?Q      Y@Y      4@a      T@q	�!i�	1@y��]��C�?"�
both�Your program is POTENTIALLY input-bound because 17.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�38.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�17.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 