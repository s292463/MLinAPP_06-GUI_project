� 	,.��M�@,.��M�@!,.��M�@	3|�m=G @3|�m=G @!3|�m=G @"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL,.��M�@C7���?18>[G�?A���FXT�?I[rPB@Y���.���?rEagerKernelExecute 0*��ʡE[�@)      @=2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map���S�?!��.#YI@)��9z��?1�Urw>G@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�U��L�?!V���B@)GV~��?1 QT#o@@:Preprocessing2F
Iterator::Model��c!:�?!�����X!@)^d~�$�?1s6�P5@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeata��M��?!Y��m$@)F@�#H��?1���D�@:Preprocessing2U
Iterator::Model::ParallelMapV2�.���Ǖ?!�ԝ�@)�.���Ǖ?1�ԝ�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��M�E�?!{�g���@)�YL��?1�6�\)Q@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�W歺�?!<>i�\F@)������?1�?U���?:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch/4�i���?!^�.W6��?)/4�i���?1^�.W6��?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��O�m�?!-n���PD@)�v��~?1<a-���?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��O���x?!y��E��?)��O���x?1y��E��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice0-��as?!z.�o��?)0-��as?1z.�o��?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range�H�+�p?!�I��?)�H�+�p?1�I��?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenatem�Yg|_|?!���Q�?)z�ަ?�a?1�x��&P�?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensorh�N?��T?!�gU�)��?)h�N?��T?1�gU�)��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 21.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�49.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no92|�m=G @I5�Bf�Q@Q�;>��:@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	C7���?C7���?!C7���?      ��!       "	8>[G�?8>[G�?!8>[G�?*      ��!       2	���FXT�?���FXT�?!���FXT�?:	[rPB@[rPB@![rPB@B      ��!       J	���.���?���.���?!���.���?R      ��!       Z	���.���?���.���?!���.���?b      ��!       JGPUY2|�m=G @b q5�Bf�Q@y�;>��:@�"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGradq�mX���?!q�mX���?"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits�S1�t�?!���DB��?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFiltery�v˨?!����t�?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput�P��\�?!2�b��K�?0"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGradV�'����?!�l�cr�?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradnX�2�A�?!��Xf��?"1
model/Conv1D_2/conv1dConv2D{��{b1�?!�� ����?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter9��0���?!5�ҿ�?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterTD�sc�?!JBW	��?0"1
model/Conv1D_3/conv1dConv2D��H�?!\��G���?Q      Y@YExR��y5@a�a�ۀ�S@qO �v��2@y�U�С�?"�
both�Your program is POTENTIALLY input-bound because 21.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�49.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�18.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 