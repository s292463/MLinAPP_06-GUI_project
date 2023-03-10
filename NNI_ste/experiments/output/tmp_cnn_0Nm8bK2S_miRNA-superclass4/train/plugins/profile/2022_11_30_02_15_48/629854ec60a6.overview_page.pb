�	@Qٰ�UR@@Qٰ�UR@!@Qٰ�UR@	K����a�?K����a�?!K����a�?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC@Qٰ�UR@.�;1���?1a��*,@I�G,7M@Y��d��~�?rEagerKernelExecute 0*	��(\�~k@2F
Iterator::Model�d�u�?!r�d*m�K@)&��:���?1�|B��|=@:Preprocessing2U
Iterator::Model::ParallelMapV2��_���?!eN���a:@)��_���?1eN���a:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�A����?!����:5@))��/���?1�m�#�D2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice��n�Uf�?!�7Y+q'@)��n�Uf�?1�7Y+q'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��x@ٸ?!��ՒF@)t(CUL��?1ډaѬ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�&��?!Gm��̿/@)m�)嵂?1��%AC�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensoriSu�l�z?!.qQ�.�@)iSu�l�z?1.qQ�.�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�79.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9K����a�?I~�w&T@Q'�љ 3@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	.�;1���?.�;1���?!.�;1���?      ��!       "	a��*,@a��*,@!a��*,@*      ��!       2      ��!       :	�G,7M@�G,7M@!�G,7M@B      ��!       J	��d��~�?��d��~�?!��d��~�?R      ��!       Z	��d��~�?��d��~�?!��d��~�?b      ��!       JGPUYK����a�?b q~�w&T@y'�љ 3@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter-z��d��?!-z��d��?0"1
model/Conv1D_2/conv1dConv2D�dWr�?!������?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput�P}~4m�?!LOG��?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterγqж�?!ƅU���?0"1
model/Conv1D_3/conv1dConv2D��1�?!I	�8ٳ�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad{Տ)��?!�K['��?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad�����?!���_6�?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter8�`�-�?!�Z����?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput�	����?!��޸� �?0"\
=model/Conv1D_1/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose��Q+�?!J�mK��?Q      Y@Y=P9��_)@a�՘H�U@qM&m.�@y�ͭ��/�?"�

device�Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�79.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 