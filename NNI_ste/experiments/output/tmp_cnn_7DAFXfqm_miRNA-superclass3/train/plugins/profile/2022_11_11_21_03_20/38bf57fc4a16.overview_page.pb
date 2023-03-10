�	��@�>@��@�>@!��@�>@	��L_@��L_@!��L_@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��@�>@�]K�}�?1v8�Jw�@A���Y��?Iw���V@Y�V'g(�?rEagerKernelExecute 0*	�K7�AZq@2U
Iterator::Model::ParallelMapV2���6���?!��0^8�L@)���6���?1��0^8�L@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��W�?!a��2 �-@),��NG�?1P��L�N(@:Preprocessing2F
Iterator::Model�a�'֩�?!�E)v�YQ@)'h��'��?1C��8�_'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipꗈ�ο�?!w�Z'��>@)�P����?1�[?��L@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�O �Ȓ�?!��޴y�!@)�1��|�?1>4ݙ�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�mm�y��?!�숌YY@)�mm�y��?1�숌YY@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor|,G�@~?!E0�;H@)|,G�@~?1E0�;H@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�j�TQ�?!�,VE��#@)�N^�e?1V�z���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 19.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�43.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��L_@I�&k��O@Q��҇t@@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�]K�}�?�]K�}�?!�]K�}�?      ��!       "	v8�Jw�@v8�Jw�@!v8�Jw�@*      ��!       2	���Y��?���Y��?!���Y��?:	w���V@w���V@!w���V@B      ��!       J	�V'g(�?�V'g(�?!�V'g(�?R      ��!       Z	�V'g(�?�V'g(�?!�V'g(�?b      ��!       JGPUY��L_@b q�&k��O@y��҇t@@�"1
model/Conv1D_2/conv1dConv2DBm �H�?!Bm �H�?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��B[,�?!�1�Q��?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilteri���d�?!5Kn��:�?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputzL�6�&�?!*oE��?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputk�V�;�?!F�����?0"1
model/Conv1D_3/conv1dConv2D&�:���?!��疛��?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad���^��?!��!�1k�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradď��s��?!�����?"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits���š0�?!v�	���?"K
$Adam/Adam/update_8/ResourceApplyAdamResourceApplyAdam�� ���?! �I����?Q      Y@YsO#,�4*@a�{a�U@qv���;@yG7}�}�?"�
both�Your program is POTENTIALLY input-bound because 19.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�43.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�27.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 