�	D� �y,@D� �y,@!D� �y,@	�h���?�h���?!�h���?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLD� �y,@��n�oj@1�M�#��?A��%s,�?I�:��@Y��x#�?rEagerKernelExecute 0*	5^�I�`@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatQS�'��?!�����?@))=�K�e�?1ZD#x��:@:Preprocessing2F
Iterator::Model���G��?!B�	�C@)����뉞?1��� T6@:Preprocessing2U
Iterator::Model::ParallelMapV2ė�"�n�?!�h��!1@)ė�"�n�?1�h��!1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�̒ 5��?!U���l�4@)�r�4��?1��G?^+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���� ��?!'��39@)���� ��?1'��39@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip9��!��?!���X�DN@)�I����?1�2��ڀ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor<���ܴy?!�X�Ӥ�@)<���ܴy?1�X�Ӥ�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapY�yVҊ�?!D�*O�7@)�5��f?1w�+N� @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 53.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�37.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�h���?I��ièV@Qաj�m[!@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��n�oj@��n�oj@!��n�oj@      ��!       "	�M�#��?�M�#��?!�M�#��?*      ��!       2	��%s,�?��%s,�?!��%s,�?:	�:��@�:��@!�:��@B      ��!       J	��x#�?��x#�?!��x#�?R      ��!       Z	��x#�?��x#�?!��x#�?b      ��!       JGPUY�h���?b q��ièV@yաj�m[!@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�	]x�?!�	]x�?0"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogitsL_ ��!�?!0��ͻ?"1
model/Conv1D_2/conv1dConv2Do���8�?!�t�����?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputHg�|�ӡ?!��7��a�?0"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput�m��fd�?!�M�z�?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad��+�8|�?!�g�`/*�?"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter=��\���?!5�)F�d�?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput��3�ՙ?!s�g	�?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterz��fș?!+�r���?0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad`�o�?!��EC�5�?Q      Y@Y�JG�(@a��7a�U@q��nXG@y���D�v�?"�
both�Your program is POTENTIALLY input-bound because 53.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�37.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�46.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 