�	�^~�Ɍ@�^~�Ɍ@!�^~�Ɍ@	��\�!$@��\�!$@!��\�!$@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�^~�Ɍ@�	m9��?175�|�]�?A,��� �?IL�4�ǡ@Y;������?rEagerKernelExecute 0*	��ʡa@2F
Iterator::Model\Y���"�?!
}/�5G@)��m��?1���� >@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatS@�� k�?!�W���>@)�Q���?1�7��9@:Preprocessing2U
Iterator::Model::ParallelMapV2�U��;M�?!�p�2��/@)�U��;M�?1�p�2��/@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceL�����?!�j���X @)L�����?1�j���X @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�V`��?!���#��J@)ѯ����?1>�t@�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate0�70�Q�?!�_�.��,@)�����?1>陹�E@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;s	��{?!�FN�@);s	��{?1�FN�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��ͪ�Ֆ?!qbBI0@)�B�Գ d?1�ܵ�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 5.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�48.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2t11.4 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9��\�!$@I�cZ�\�M@Qa�,cA@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�	m9��?�	m9��?!�	m9��?      ��!       "	75�|�]�?75�|�]�?!75�|�]�?*      ��!       2	,��� �?,��� �?!,��� �?:	L�4�ǡ@L�4�ǡ@!L�4�ǡ@B      ��!       J	;������?;������?!;������?R      ��!       Z	;������?;������?!;������?b      ��!       JGPUY��\�!$@b q�cZ�\�M@ya�,cA@�"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�t���?!�t���?0"1
model/Conv1D_3/conv1dConv2D;^N �^�?!:��iz�?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter΋b��?!
g�*��?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputx��l�?!��5�BQ�?0"1
model/Conv1D_2/conv1dConv2D^��֜8�?!`r�V��?"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGradى~�3�?!������?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput�o�p*��?!x{�c�4�?0"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits�?�
S�?!UmW��?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad�ض��њ?!�����?"C
%gradient_tape/model/Conv1D_2/ReluGradReluGradZ�j�8�?!Ȋ�=�8�?Q      Y@Y�;�;-@a��؉�XU@qA� �@@y6�N��?"�
both�Your program is MODERATELY input-bound because 5.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�48.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.moderate"t11.4 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�32.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 