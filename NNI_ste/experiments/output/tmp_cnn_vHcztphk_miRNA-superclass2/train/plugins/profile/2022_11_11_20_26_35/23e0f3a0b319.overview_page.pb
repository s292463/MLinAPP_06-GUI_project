�	�(��h@�(��h@!�(��h@	2y߿�{�?2y߿�{�?!2y߿�{�?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�(��h@j��4ӽ�?1ǻ#c��d@A���2#�?I,�뇈?@Y�����?rEagerKernelExecute 0*	G�z�'`@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�Oq�?!��=��@@)Mg'���?1�49�0?@:Preprocessing2F
Iterator::Model� �K��?!��T���B@)�.���?1f��%;5@:Preprocessing2U
Iterator::Model::ParallelMapV2+~��7�?!ݮ�{=0@)+~��7�?1ݮ�{=0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��Ss���?!_$�pN^O@)H1@�	�?1ij�\1�)@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeata��_Yi�?!�P���+@)�u�!H�?1��W��!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��-</{?!��]A�v@)��-</{?1��]A�v@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�}�e�ħ?!�)o[��A@)�@fg�;e?1�@�h @:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensorr�&"�`?!H{����?)r�&"�`?1H{����?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice.����W?!_��W�?).����W?1_��W�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�16.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no91y߿�{�?ILrt��0@Q~|K���T@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	j��4ӽ�?j��4ӽ�?!j��4ӽ�?      ��!       "	ǻ#c��d@ǻ#c��d@!ǻ#c��d@*      ��!       2	���2#�?���2#�?!���2#�?:	,�뇈?@,�뇈?@!,�뇈?@B      ��!       J	�����?�����?!�����?R      ��!       Z	�����?�����?!�����?b      ��!       JGPUY1y߿�{�?b qLrt��0@y~|K���T@�"1
model/Conv1D_3/conv1dConv2D�.�g��?!�.�g��?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput|��֗�?!��wf���?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�C�uI�?!䳥����?0"1
model/Conv1D_4/conv1dConv2D~1���?!���!���?"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilterx\%k���?!l9����?0"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInputڀ����?!z��!T�?0"1
model/Conv1D_2/conv1dConv2D�s�f
��?!�c���?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterE�<Z(�?!\�.�$�?0"1
model/Conv1D_1/conv1dConv2D���m��?!���4_s�?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputv$Cf1>�?!�HKB��?0Q      Y@Y�F��@a4�돗�W@q���AF@y�{ܬe?"�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�16.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�44.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 