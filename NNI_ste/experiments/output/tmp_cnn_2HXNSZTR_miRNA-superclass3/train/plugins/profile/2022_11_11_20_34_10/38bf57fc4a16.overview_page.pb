�	��h>�2@��h>�2@!��h>�2@	��8���@��8���@!��8���@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��h>�2@ᶶ�d@1��ID�7@Aυ�^��?IT6��,� @Yٲ|]���?rEagerKernelExecute 0*	��Mb�d@2U
Iterator::Model::ParallelMapV2ެ����?!p���*C8@)ެ����?1p���*C8@:Preprocessing2F
Iterator::Model>\r�)�?!����G@)b�7�W��?18��u2�6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��{�?! 1�L0;@)zUg���?1��er6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenatenO����?!|�@m��2@)��F��?1J o��c'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice*�"��?!\e%���@)*�"��?1\e%���@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip� \�z�?!,��~QcJ@)�GnM�-�?1�cJ�*@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorV��Dׅ?!�͚��@)V��Dׅ?1�͚��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapI�V���?!a���Ω4@)��d��g?1F>7��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 37.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�45.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��8���@I:�)��T@Q|3I��-@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	ᶶ�d@ᶶ�d@!ᶶ�d@      ��!       "	��ID�7@��ID�7@!��ID�7@*      ��!       2	υ�^��?υ�^��?!υ�^��?:	T6��,� @T6��,� @!T6��,� @B      ��!       J	ٲ|]���?ٲ|]���?!ٲ|]���?R      ��!       Z	ٲ|]���?ٲ|]���?!ٲ|]���?b      ��!       JGPUY��8���@b q:�)��T@y|3I��-@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��/���?!��/���?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?�L����?!2�&g�?0"1
model/Conv1D_2/conv1dConv2DpPz�%J�?!5��lF�?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�t�5_�?!��~S�h�?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad=i!8ڟ?!d���+f�?"1
model/Conv1D_3/conv1dConv2D�0�g&=�?!m<:�)�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad�6�k�?!ܘ�����?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput{� #䶙?!��<(|�?0"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits
�&?��?!������?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�y��?!u�Z(�l�?0Q      Y@Y�a�2�t'@a�ӭ�aV@q���6��L@yۜ�����?"�
both�Your program is POTENTIALLY input-bound because 37.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�45.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�57.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 