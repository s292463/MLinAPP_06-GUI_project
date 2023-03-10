�	�{��@�{��@!�{��@	X�%5Q��?X�%5Q��?!X�%5Q��?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�{��@9���	�?1U�3�9 @A�O:�`��?I��_̖�@Y� ����?rEagerKernelExecute 0*	_�I�d@2F
Iterator::Model�V�����?!����H@)�.��?1d$oW�@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat5���#�?!��.�>e>@)�yrM�̦?1˦B���:@:Preprocessing2U
Iterator::Model::ParallelMapV25)�^Ҙ?!�*���,@)5)�^Ҙ?1�*���,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�9�m½�?!l�#��%@)�9�m½�?1l�#��%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipS�
cA�?!Y0�u�I@) �C��<~?1�Lj�U�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(
�<I�?!��P���.@)�A�p�-~?1�n��̋@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��-��z?!�6_�@)��-��z?1�6_�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���D-�?!�?g��0@)�Ŧ�B g?1���[��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 23.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�47.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9W�%5Q��?IV�Ny�Q@Q橒�3:@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	9���	�?9���	�?!9���	�?      ��!       "	U�3�9 @U�3�9 @!U�3�9 @*      ��!       2	�O:�`��?�O:�`��?!�O:�`��?:	��_̖�@��_̖�@!��_̖�@B      ��!       J	� ����?� ����?!� ����?R      ��!       Z	� ����?� ����?!� ����?b      ��!       JGPUYW�%5Q��?b qV�Ny�Q@y橒�3:@�"1
model/Conv1D_4/conv1dConv2D�ǚ3���?!�ǚ3���?"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�`�Eū?!��}� $�?0"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad��pݥ?! �Cp	�?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterϘқ���?!4�Ϫ]6�?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter���󖋠?!���gCY�?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad���3j�?!��4�D��?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputa�Q�ʕ�?!q����?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput*u'��1�?!�9<�o�?0"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput~ו�/�?!<���AP�?0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad����˜?!��w��?Q      Y@Y@n]�G*@a8R4��U@q��
f�FR@yҵ�V7}�?"�
both�Your program is POTENTIALLY input-bound because 23.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�47.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�73.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 