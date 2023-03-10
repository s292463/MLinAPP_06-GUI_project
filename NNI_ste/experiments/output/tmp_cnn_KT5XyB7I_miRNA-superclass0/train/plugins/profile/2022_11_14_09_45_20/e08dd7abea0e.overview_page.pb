�	�|��z$@�|��z$@!�|��z$@	��P$�D @��P$�D @!��P$�D @"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�|��z$@W$&���@1w� ݗS@AR~R���?I�o��@Y��H�+�?rEagerKernelExecute 0*	 ��Q��a@2U
Iterator::Model::ParallelMapV2��ŉ�v�?!I׃�-<@)��ŉ�v�?1I׃�-<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatܜJ�*�?!LC���>@)��X6sH�?1�N� +-9@:Preprocessing2F
Iterator::Model��:TS��?!c�
`2H@)��_=�[�?1}`��64@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice ��c�?!���$�@) ��c�?1���$�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenatebHN&n�?!��=���+@)���x#�?1�����Z@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�ِf?!�ң9cc@)�ِf?1�ң9cc@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip-�2���?!�y����I@)tϺFˁ~?1��D3@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap׆�q��?!SN�ީ/@)�u�!Hg?1�IC � @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 8.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�44.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t23.4 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9��P$�D @I
ą��P@Q�k��y8@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	W$&���@W$&���@!W$&���@      ��!       "	w� ݗS@w� ݗS@!w� ݗS@*      ��!       2	R~R���?R~R���?!R~R���?:	�o��@�o��@!�o��@B      ��!       J	��H�+�?��H�+�?!��H�+�?R      ��!       Z	��H�+�?��H�+�?!��H�+�?b      ��!       JGPUY��P$�D @b q
ą��P@y�k��y8@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterٛ�?�'�?!ٛ�?�'�?0"1
model/Conv1D_2/conv1dConv2D&��Y�:�?!�̼�U1�?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputH<�`��?!����n�?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterY&Wc�Ĩ?!(��}:��?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad�;U_�?!�pH�{�?"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits�q8��y�?!����?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad
sR�&�?!t�y���?"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad6e&���?!�3�
p�?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�.�`͝?!�����L�?0"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose*3���?!�i����?Q      Y@Y|�^���-@aQ(
�BU@q�gW7@y*
����?"�
both�Your program is MODERATELY input-bound because 8.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�44.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t23.4 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�23.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 