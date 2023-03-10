�	P�R)@P�R)@!P�R)@	�vʝa|	@�vʝa|	@!�vʝa|	@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLP�R)@dY0�G�?1uYLl>�@A�71$'�?I��a��?Y�ٕ���?rEagerKernelExecute 0*	t�V.r@2U
Iterator::Model::ParallelMapV2DP5z5@�?!0�n��I@)DP5z5@�?10�n��I@:Preprocessing2F
Iterator::Modeld�� w�?!�'1y��Q@)��M�E�?1ۮ��-O2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat$)�ahu�?!�֯���1@)�Y�>�-�?1���; /@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�E���Ԉ?!�i+1�@)�E���Ԉ?1�i+1�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate&�fe���?![��J�@)����L�?1g�\s3�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip5���:U�?!�`;J�=@)�ݳ�т?1����XE	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor|~!<z?!��ڪR�@)|~!<z?1��ڪR�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���B�?!��M�!"@)@L<�k?1�A��Fx�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 15.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�28.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�vʝa|	@I}Fgt
F@Q���]J@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	dY0�G�?dY0�G�?!dY0�G�?      ��!       "	uYLl>�@uYLl>�@!uYLl>�@*      ��!       2	�71$'�?�71$'�?!�71$'�?:	��a��?��a��?!��a��?B      ��!       J	�ٕ���?�ٕ���?!�ٕ���?R      ��!       Z	�ٕ���?�ٕ���?!�ٕ���?b      ��!       JGPUY�vʝa|	@b q}Fgt
F@y���]J@�"1
model/Conv1D_2/conv1dConv2DNVyɺ�?!NVyɺ�?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��'��R�?!��P`��?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��X��?!u����?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput;���S�?!B�a�}��?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput�~ƹ�?!"���?0"1
model/Conv1D_3/conv1dConv2DҭO�*V�?!�y׎���?"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFiltercX�"l��?!z Q>g�?0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad�kk�?!zP 	��?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad-NTeǚ?!]�@^k��?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad�%�?!ks�KI�?Q      Y@Y      )@a     �U@qUF�_�=@y��4 }�?"�
both�Your program is POTENTIALLY input-bound because 15.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�28.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�29.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 