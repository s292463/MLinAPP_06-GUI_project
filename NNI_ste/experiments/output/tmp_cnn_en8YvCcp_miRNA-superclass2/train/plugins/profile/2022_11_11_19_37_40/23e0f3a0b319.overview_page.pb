� 	��-u�@��-u�@!��-u�@	L���� @L���� @!L���� @"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��-u�@��h�xT�?1���d$@A�'�y��?I�-��ĝ@YDkE����?rEagerKernelExecute 0*	Zd;_��@2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Mapx~Q����?!�=���K@)�DkE��?1���7J@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��ZC�=�?!|ߩ_�D@)��[�d��?1�I��URC@:Preprocessing2F
Iterator::Model�C�H���?!�.��'
@)]��7���?1��b���?:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat�+�p�{�?!���y�o�?)E�u����?1�v�?
��?:Preprocessing2U
Iterator::Model::ParallelMapV2���!6X�?!~�����?)���!6X�?1~�����?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice�"h�$�?!u,�"��?)�"h�$�?1u,�"��?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate����rێ?!�0���=�?)�|?q �?1\��>ש�?:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�SH�9�?!@r�PS3�?)��DJ�y�?1#Z�$A7�?:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch���o�?!l>��dZ�?)���o�?1l>��dZ�?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip)�1k�?!g���ZE@)~t��gy~?1���>j��?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����w?!�a��^�?)����w?1�a��^�?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangevnڌ�u?!S�� ��?)vnڌ�u?1S�� ��?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�c]�F�?!�%A�k�?)K��`?1'�?����?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor[�a/�M?!ۄ�M4?�?)[�a/�M?1ۄ�M4?�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 14.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�34.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9L���� @I��I�.�H@Qg}�zG@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��h�xT�?��h�xT�?!��h�xT�?      ��!       "	���d$@���d$@!���d$@*      ��!       2	�'�y��?�'�y��?!�'�y��?:	�-��ĝ@�-��ĝ@!�-��ĝ@B      ��!       J	DkE����?DkE����?!DkE����?R      ��!       Z	DkE����?DkE����?!DkE����?b      ��!       JGPUYL���� @b q��I�.�H@yg}�zG@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter����I�?!����I�?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad�FI]N�?!����u8�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad�� �Ť?!bOג�i�?"1
model/Conv1D_2/conv1dConv2D��fx�n�?!��0��?"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose���|�~�?!�H��?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�fI�.�?!�q@���?0"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose�wU}��?!O��fK�?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputy��ş?!')�!��?0"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose��˻�?!r���X�?"3
model/Conv1D_1/BiasAddBiasAdd��#��?!������?Q      Y@Yj+����2@a&���VRT@q�+#@[(@y
	On�?"�
both�Your program is POTENTIALLY input-bound because 14.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�34.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�12.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 