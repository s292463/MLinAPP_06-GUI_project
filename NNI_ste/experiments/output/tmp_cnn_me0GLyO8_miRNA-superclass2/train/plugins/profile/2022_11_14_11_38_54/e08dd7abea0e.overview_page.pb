�	(��h�\@(��h�\@!(��h�\@	L�L?�y@L�L?�y@!L�L?�y@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL(��h�\@v��ť��?1˟oK@AQ.�_x%�?I�U���@Y4���l��?rEagerKernelExecute 0*	�x�&1�r@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate&p�n��?!���,@M@)&o�����?1�ۮ�'�I@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�T�G��?!y/#�ߦ0@)^�SH�?1�׭.B�,@:Preprocessing2F
Iterator::Model�z�G�?!�mm=�H1@)`��橞?1zʫ�Q�#@:Preprocessing2U
Iterator::Model::ParallelMapV2b�qm��?!	#^&m�@)b�qm��?1	#^&m�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�}���?!W��0@)�}���?1W��0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip6�.6��?!����ޭT@)Y���RA�?1K�JOU@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorе/��|?!]b���@)е/��|?1]b���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�+f���?!�g\��M@)�j�=&Rj?1�&��|��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 6.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�40.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t21.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9M�L?�y@I1���~#O@Q�:�Ꝛ?@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	v��ť��?v��ť��?!v��ť��?      ��!       "	˟oK@˟oK@!˟oK@*      ��!       2	Q.�_x%�?Q.�_x%�?!Q.�_x%�?:	�U���@�U���@!�U���@B      ��!       J	4���l��?4���l��?!4���l��?R      ��!       Z	4���l��?4���l��?!4���l��?b      ��!       JGPUYM�L?�y@b q1���~#O@y�:�Ꝛ?@�"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad�R�\�4�?!�R�\�4�?"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogitsy�ۭ~�?!�%���ü?"1
model/Conv1D_4/conv1dConv2D����u�?!C�D�a�?"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilterπ���?!w�?�"�?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput��>���?!��РM�?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter���?!�

�*�?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput{�����?!Z���#��?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterO��4鑠?!�Qa��?0"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�����o�?!bO�_��?0"1
model/Conv1D_2/conv1dConv2Dm��#]��?!�ю��p�?Q      Y@Y�ܺ�+@a�p�h�U@q{�T:8@yk,�JP��?"�
both�Your program is MODERATELY input-bound because 6.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�40.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t21.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�24.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 