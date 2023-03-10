�	Y�n}��@Y�n}��@!Y�n}��@	�Xp��?�Xp��?!�Xp��?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLY�n}��@N��}�?1��=@A�BY��Z�?IZ��!ŨM@Y��|��	@rEagerKernelExecute 0*	i��|?�b@2F
Iterator::Model	3�z��?!��L��I@)�	�O���?1E+�N@@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�=�-�?!��w�6E>@)�Hg`�e�?1��� ��;@:Preprocessing2U
Iterator::Model::ParallelMapV2�T�]�?!J5z��2@)�T�]�?1J5z��2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�B]?!?��Q�1'@)KZ��φ?1+�|I��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��F����?!:��p2H@)��]��?1���.�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�%jjy?!Q5�Y�@)�%jjy?1Q5�Y�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapya�X5�?!樴�V�?@)�3�ۃ`?1��ͳ���?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensork*��.�^?!D�co���?)k*��.�^?1D�co���?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice��E��\Z?!����/7�?)��E��\Z?1����/7�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�10.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�Xp��?I�01ឺ%@Q�)���$V@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	N��}�?N��}�?!N��}�?      ��!       "	��=@��=@!��=@*      ��!       2	�BY��Z�?�BY��Z�?!�BY��Z�?:	Z��!ŨM@Z��!ŨM@!Z��!ŨM@B      ��!       J	��|��	@��|��	@!��|��	@R      ��!       Z	��|��	@��|��	@!��|��	@b      ��!       JGPUY�Xp��?b q�01ឺ%@y�)���$V@�"1
model/Conv1D_2/conv1dConv2D ׫q�?! ׫q�?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput��E>�?!h٭�]��?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput�)�?7�?!�����?0"1
model/Conv1D_3/conv1dConv2D�C�N�?!"f���?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��sr��?!h��n�?0"1
model/Conv1D_4/conv1dConv2D����?!Tz��Z��?"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput�׵y�φ?!�Qr՚��?0"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�&y_<�?!M6�*�1�?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�M3���?!���z�?0"1
model/Conv1D_1/conv1dConv2D�>��|?!�����?Q      Y@Y�8D�@atl�~�W@qd���F�@y�����.E?"�

device�Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 