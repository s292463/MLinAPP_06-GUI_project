�	���#�@���#�@!���#�@	xc��;
@xc��;
@!xc��;
@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL���#�@p%;6��?1�����@A������?IP��� @Yc{-�1�?rEagerKernelExecute 0*	-����a@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�#��:�?!� ����?@)`��s�?1��-�>;@:Preprocessing2F
Iterator::Model6��Ϸ�?!u d@^-D@)�o�[t�?1nj�C9@:Preprocessing2U
Iterator::Model::ParallelMapV2��-�熖?!�����.@)��-�熖?1�����.@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�磌� �?!S�p�|)+@)�磌� �?1S�p�|)+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate����A�?!�/��4@)dT8��?1Y�گU�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip.�R\U��?!�ߛ���M@)�X�|^�?1����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor	���?Qy?!�VA*{0@)	���?Qy?1�VA*{0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�TގpZ�?!�ܵ��46@)t|�8c�c?1�lm�ϛ�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 20.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�43.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9xc��;
@I�mZ��O@Q��[�r@@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	p%;6��?p%;6��?!p%;6��?      ��!       "	�����@�����@!�����@*      ��!       2	������?������?!������?:	P��� @P��� @!P��� @B      ��!       J	c{-�1�?c{-�1�?!c{-�1�?R      ��!       Z	c{-�1�?c{-�1�?!c{-�1�?b      ��!       JGPUYxc��;
@b q�mZ��O@y��[�r@@�"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilterhC��C��?!hC��C��?0"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad[�r��h�?!��f�d��?"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits�]\)��?!�_���?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��{�.�?!�i��T	�?0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad�QCsU��?!��9���?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput�Ж2c�?!�@c���?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputTϩ�)�?!�1 i���?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterr~��?!W3��ے�?0"1
model/Conv1D_3/conv1dConv2DkT�k8��?!�H�g�,�?"1
model/Conv1D_4/conv1dConv2D�5!}d��?!�[q���?Q      Y@Y&���[,@a��_��tU@qݘ�$�WA@yƛ�P��?"�
both�Your program is POTENTIALLY input-bound because 20.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�43.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�34.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 