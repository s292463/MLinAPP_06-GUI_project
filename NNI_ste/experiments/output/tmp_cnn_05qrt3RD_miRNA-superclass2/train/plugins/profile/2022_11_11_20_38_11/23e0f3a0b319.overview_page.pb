� 	��tZ��@��tZ��@!��tZ��@	��c��?��c��?!��c��?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��tZ��@��J?�,�?176;R}@Av4����?I4d<J%�@Y��'���?rEagerKernelExecute 0*	b��"�v�@2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map:�w���?!�<�B��H@)�F����?1��=�cOF@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap}A	]�?!Q`�i��D@)�8b->�?1!��U��B@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat$}ZEh�?!�h��d�@)稣�j�?1��Ub&@:Preprocessing2F
Iterator::Modelp��;��?!r�|���@)p�x�0D�?1J[���
@:Preprocessing2U
Iterator::Model::ParallelMapV2q!��Fʖ?!�3�PB@)q!��Fʖ?1�3�PB@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate}�H�F��?!8�k��r�?)	�^)ː?1����a��?:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�e��@�?![��m@)ڨN���?1{��ة�?:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch��ԱJ�?!`�����?)��ԱJ�?1`�����?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��F�?!:*��@F@)��X32�}?1�H�fqy�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor\��b��w?!v�t�"�?)\��b��w?1v�t�"�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice��A�fu?!liymS�?)��A�fu?1liymS�?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range�`��o?! ���)P�?)�`��o?1 ���)P�?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate#�tu�b{?!��\$'X�?).����W?1�T��NG�?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor��V���L?!f*~�au�?)��V���L?1f*~�au�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 16.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�39.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��c��?I���<�/L@Q�=��F�D@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��J?�,�?��J?�,�?!��J?�,�?      ��!       "	76;R}@76;R}@!76;R}@*      ��!       2	v4����?v4����?!v4����?:	4d<J%�@4d<J%�@!4d<J%�@B      ��!       J	��'���?��'���?!��'���?R      ��!       Z	��'���?��'���?!��'���?b      ��!       JGPUY��c��?b q���<�/L@y�=��F�D@�"1
model/Conv1D_2/conv1dConv2D��Y~Hv�?!��Y~Hv�?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter���u��?!00'����?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput��f�S8�?!!�@��?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradc����e�?!ʷ�:6�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradA�	�?!�,�����?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter� PWu��?!�0v��I�?0"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose�xO0�>�?!��E�Q�?"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose��-��7�?!h�e��X�?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�F�����?!ԣorT�?0"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transposej���ȸ�?!�/���O�?Q      Y@Y�?�?�4@ap�p�S@q�rV$��5@yk��a�?"�
both�Your program is POTENTIALLY input-bound because 16.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�39.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�21.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 