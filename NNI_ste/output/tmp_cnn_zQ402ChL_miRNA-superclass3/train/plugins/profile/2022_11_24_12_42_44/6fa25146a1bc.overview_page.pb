�	�����:~@�����:~@!�����:~@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC�����:~@:��K�?1P��n�ow@A�;��.�?I�o�^}�Z@rEagerKernelExecute 0*	m�����r@2F
Iterator::ModelT���=�?!�7k�]�R@)��*ø�?1Go�L�P@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate E�����?!;�ۨ>p.@)�MF�aܥ?1�|GK,@:Preprocessing2U
Iterator::Model::ParallelMapV2������?!�D��8 @)������?1�D��8 @:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�	�i��?![%���@)�9?�q��?1-E_b�P@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��Ր�ǲ?!� S�N8@)����Mb�?1��UP�4@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorJ�y�w?!f!A��?)J�y�w?1f!A��?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�x` �?!���-0@)eo)狽g?1��/
��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice�ِf_?! �4�O�?)�ِf_?1 �4�O�?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor�N^�U?!�_}�Rj�?)�N^�U?1�_}�Rj�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�22.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI _�t�x6@Q8h���aS@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	:��K�?:��K�?!:��K�?      ��!       "	P��n�ow@P��n�ow@!P��n�ow@*      ��!       2	�;��.�?�;��.�?!�;��.�?:	�o�^}�Z@�o�^}�Z@!�o�^}�Z@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q _�t�x6@y8h���aS@�"1
model/Conv1D_2/conv1dConv2D�[��,�?!�[��,�?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��J�s��?!H��nP��?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput�Vj�2J�?!�Ѩ�i��?0"1
model/Conv1D_3/conv1dConv2D�|�EP�?!SY�.��?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter �6��?!T�h%�?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputf����?!w����?0"1
model/Conv1D_4/conv1dConv2D;����?!Y���K�?"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput��<��?!�0��&��?0"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilterPc*��?!!�8D�$�?0"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilterE�@�H��?!��%hP��?0Q      Y@YN��N�D@a;�;��W@q�:Q3�B@y��$R?"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�22.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�37.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 