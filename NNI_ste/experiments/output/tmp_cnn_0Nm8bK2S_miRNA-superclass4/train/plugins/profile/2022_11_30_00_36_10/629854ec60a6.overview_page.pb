�	Qi��>�:@Qi��>�:@!Qi��>�:@	��L֜[�?��L֜[�?!��L֜[�?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLQi��>�:@�m��*@1����%@Aǜg�K6�?I�2�}�E�?Y���tw��?rEagerKernelExecute 0*	7�A`�c@2F
Iterator::Model�5��Wt�?!����iAH@))�7Ӆ�?1����>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatL<��?!�vv��M?@)�z���?1{�9"�r;@:Preprocessing2U
Iterator::Model::ParallelMapV2 UܸŜ?!1���1@) UܸŜ?11���1@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�"j��G�?!�Z��@)�"j��G�?1�Z��@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�\QJV�?!d�����*@) �8�@d�?1Ŵ����@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�� w��?!!	;��I@)�u��$�?1P��S @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorm:�Y�x?!�q�9�@)m:�Y�x?1�q�9�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�c#��?!nki���-@)D6�.6�d?1MX�>~��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 50.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�7.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��L֜[�?I��6x�M@Q5���TD@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�m��*@�m��*@!�m��*@      ��!       "	����%@����%@!����%@*      ��!       2	ǜg�K6�?ǜg�K6�?!ǜg�K6�?:	�2�}�E�?�2�}�E�?!�2�}�E�?B      ��!       J	���tw��?���tw��?!���tw��?R      ��!       Z	���tw��?���tw��?!���tw��?b      ��!       JGPUY��L֜[�?b q��6x�M@y5���TD@�"1
model/Conv1D_2/conv1dConv2DV��;砸?!V��;砸?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�R��	�?!��.����?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput�8�{�?!��|����?0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad9�W�q��?!GIr���?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter\�q�c�?!�	��aI�?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad]a~��?!�5Q�j�?"1
model/Conv1D_3/conv1dConv2D�/�s��?!��?��?"3
model/Conv1D_1/BiasAddBiasAdd��5���?!kzI�
[�?"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose��ppl�?!u�PA�!�?"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transposei��8�?!�2X��?Q      Y@Y&W�+�)@a�����U@q����1@y���~vG�?"�
both�Your program is POTENTIALLY input-bound because 50.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�7.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�17.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 