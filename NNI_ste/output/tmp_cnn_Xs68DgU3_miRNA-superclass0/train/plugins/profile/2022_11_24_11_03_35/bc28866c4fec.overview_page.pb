�	��{�_@��{�_@!��{�_@	�Ѿ�?@�Ѿ�?@!�Ѿ�?@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��{�_@�(z�c��?1l�u�b@A�k_@/ܙ?I ��W@Yʊ�� ��?rEagerKernelExecute 0*	NbX9xf@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateߨ��5�?!V�]�A@)�}U.T��?1�K_1dK@@:Preprocessing2F
Iterator::Modelӄ�'c|�?!Cď7BF@)�l:�Y�?1!ļQ��>@:Preprocessing2U
Iterator::Model::ParallelMapV2�9A�>�?!�ؖ�Qm+@)�9A�>�?1�ؖ�Qm+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��L�*��?!��;pȽK@)�|�H�F�?1%��P�!@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�m��)�?!Zf/q�#@)�Rb���?1����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor`�Eжz?!X�O5�@)`�Eжz?1X�O5�@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor���0i?!����U:�?)���0i?1����U:�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�!�A�?!]귗'cB@)q!��F�f?1�`F'��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice.c}�[?!�����?).c}�[?1�����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 16.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�38.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�Ѿ�?@I[{�n��K@Q�r^H�D@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�(z�c��?�(z�c��?!�(z�c��?      ��!       "	l�u�b@l�u�b@!l�u�b@*      ��!       2	�k_@/ܙ?�k_@/ܙ?!�k_@/ܙ?:	 ��W@ ��W@! ��W@B      ��!       J	ʊ�� ��?ʊ�� ��?!ʊ�� ��?R      ��!       Z	ʊ�� ��?ʊ�� ��?!ʊ�� ��?b      ��!       JGPUY�Ѿ�?@b q[{�n��K@y�r^H�D@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter���ջN�?!���ջN�?0"1
model/Conv1D_2/conv1dConv2D$n�om�?!f~ׯ^�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad[B*�}��?!��5��?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad# �[�F�?!���֘�?"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose��-��4�?!Vk�S�?"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose�#�e� �?!�J7��?"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transposeӓ����?!E���G�?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilterǍ�%Cl�?!��lwC��?0"3
model/Conv1D_1/BiasAddBiasAdd��Q��?!Q�!��?"-
model/Conv1D_1/ReluRelu��6�"�?!,(QC%V�?Q      Y@Y�?�pJ�*@a�����U@q�7+;A@yw+��?"�
both�Your program is POTENTIALLY input-bound because 16.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�38.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�34.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 