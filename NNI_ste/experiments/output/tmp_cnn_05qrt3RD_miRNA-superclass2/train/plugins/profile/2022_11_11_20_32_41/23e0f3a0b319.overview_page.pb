�	�M�M��p@�M�M��p@!�M�M��p@	�ƭԄ��?�ƭԄ��?!�ƭԄ��?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�M�M��p@?�7j�i�?1K��>7m@A�lsczº?I3�68E?@Yl|&��i�?rEagerKernelExecute 0*	o���e`@2F
Iterator::Model���y7�?!�����G@)b���4�?1=��r�>@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate@0G��ۤ?!�!nk?@)d�6��:�?1��� $;@:Preprocessing2U
Iterator::Model::ParallelMapV2�<��tZ�?! 6���b1@)�<��tZ�?1 6���b1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�5�;Nё?!'{6:�*@)�;2V���?1M�V��!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�w�~�~�?!F�WzJ@)����A~?1
��hL�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor)_�BFw?!���_S@))_�BFw?1���_S@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensorI��Z��g?!�u7��@)I��Z��g?1�u7��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?�'I�L�?!4�U=�@@)��{g?1��Vd�*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlices�m�B<b?!hcOx&�?)s�m�B<b?1hcOx&�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�11.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�ƭԄ��?IЙ?��(@Q�����U@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	?�7j�i�??�7j�i�?!?�7j�i�?      ��!       "	K��>7m@K��>7m@!K��>7m@*      ��!       2	�lsczº?�lsczº?!�lsczº?:	3�68E?@3�68E?@!3�68E?@B      ��!       J	l|&��i�?l|&��i�?!l|&��i�?R      ��!       Z	l|&��i�?l|&��i�?!l|&��i�?b      ��!       JGPUY�ƭԄ��?b qЙ?��(@y�����U@�"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput)�Qh�j�?!)�Qh�j�?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�(���X�?!Z�n���?0"1
model/Conv1D_2/conv1dConv2DY�A=kr�?!����?"1
model/Conv1D_3/conv1dConv2DT�e�;��?!�vv
D�?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter���N�?!8mB�&�?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput�% ~�[�?!fn3[^��?0"1
model/Conv1D_4/conv1dConv2DM=$��^�?!P��T��?"1
model/Conv1D_1/conv1dConv2D]�DƔu�?!۷6�X�?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter9Pep���?!]��~��?0"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInputt-�=\�?!ɋ��Q]�?0Q      Y@Y�Š"�R@a���=��W@qs�o ��F@y��ʒ�Q?"�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�45.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 