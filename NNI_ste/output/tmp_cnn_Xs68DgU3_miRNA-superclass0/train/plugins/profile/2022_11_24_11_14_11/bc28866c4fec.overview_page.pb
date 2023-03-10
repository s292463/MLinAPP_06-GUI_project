�	�);��>@�);��>@!�);��>@	E+:B�g@E+:B�g@!E+:B�g@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�);��>@Q�v0b�?1��ᔹ��?A|DL�$z�?I�_ѭW@Y�.��0�?rEagerKernelExecute 0*	Zd;ߓb@2F
Iterator::Model��Q���?!|��&
�I@)�y ����?1��b?A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat$�@�ؤ?!�jWe;@)l�V^�?�?1f,�<H�6@:Preprocessing2U
Iterator::Model::ParallelMapV2ϺFˁ�?!z�c�)1@)ϺFˁ�?1z�c�)1@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice;oc�#Շ?!bw�;�Q@);oc�#Շ?1bw�;�Q@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�J�h�?!.}[�!,@)��1���?1��� ��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��-�熲?!�]k��XH@)ҩ+��y�?1a�g�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�?�|?!��g/;�@)�?�|?1��g/;�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapZ*oG8-�?!�J�ȧ�/@)bJ$��(f?1,kNR��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 19.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�48.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9E+:B�g@I���3$Q@Q3M���U;@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Q�v0b�?Q�v0b�?!Q�v0b�?      ��!       "	��ᔹ��?��ᔹ��?!��ᔹ��?*      ��!       2	|DL�$z�?|DL�$z�?!|DL�$z�?:	�_ѭW@�_ѭW@!�_ѭW@B      ��!       J	�.��0�?�.��0�?!�.��0�?R      ��!       Z	�.��0�?�.��0�?!�.��0�?b      ��!       JGPUYE+:B�g@b q���3$Q@y3M���U;@�"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput�������?!�������?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter}�{�?!|s�^3n�?0"1
model/Conv1D_2/conv1dConv2D�M�w�:�?!"MCM��?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter������?!܆n�y��?0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad��X���?!������?"C
%gradient_tape/model/Conv1D_2/ReluGradReluGradւp��R�?!�'��`�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradI�>XƗ?!j�̠��?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter����?!���K�?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputZ����R�?!�D����?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad{U?��(�?!'��@�?Q      Y@Yyxxxxx*@a�����U@qܸ��i<@y�z��O�?"�
both�Your program is POTENTIALLY input-bound because 19.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�48.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�28.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 