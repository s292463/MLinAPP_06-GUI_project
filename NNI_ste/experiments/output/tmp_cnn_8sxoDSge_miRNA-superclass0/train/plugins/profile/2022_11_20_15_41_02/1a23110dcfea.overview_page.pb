�	l��[�#@l��[�#@!l��[�#@	�)�2��@�)�2��@!�)�2��@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLl��[�#@~�<�r�?1�}s��@A䠄���?I�R{m@Y�Z&��|�?rEagerKernelExecute 0*	��K7�d@2F
Iterator::Modelg�lt�O�?!Ƴ��tD@)      �?1��
 \<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat����*�?!�.�+/�=@)&m��ͥ?1z���u�9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate����?!�	Rw�7@)���9��?1W;�,�0@:Preprocessing2U
Iterator::Model::ParallelMapV2���9?�?!�e�SB)@)���9?�?1�e�SB)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?���?!9KŹ@)?���?19KŹ@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��� �?!:L[(�M@),��ص�}?1!��p^�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�ۡa1�z?!�(1��@)�ۡa1�z?1�(1��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapG6u�?!��T��8@)��A�Fc?1�+p���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 16.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�21.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�)�2��@Ig����B@Q�G���M@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	~�<�r�?~�<�r�?!~�<�r�?      ��!       "	�}s��@�}s��@!�}s��@*      ��!       2	䠄���?䠄���?!䠄���?:	�R{m@�R{m@!�R{m@B      ��!       J	�Z&��|�?�Z&��|�?!�Z&��|�?R      ��!       Z	�Z&��|�?�Z&��|�?!�Z&��|�?b      ��!       JGPUY�)�2��@b qg����B@y�G���M@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�� �7�?!�� �7�?0"1
model/Conv1D_2/conv1dConv2D���	H��?!��ƒ���?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput�q���Y�?!LX3R("�?0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad7�o3J �?!G�:��?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��Z��?!�Ch���?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad5��>�?!e�a����?"1
model/Conv1D_3/conv1dConv2D� ���?!~F~r���?"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose�w
T<�?!z����?"3
model/Conv1D_1/BiasAddBiasAdd"Y�kK�?!��r*��?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter���~�e�?!��Z�T��?0Q      Y@YD+l$Z)@a�z2~��U@q��vlΆ7@yA�B���?"�
both�Your program is POTENTIALLY input-bound because 16.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�21.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�23.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 