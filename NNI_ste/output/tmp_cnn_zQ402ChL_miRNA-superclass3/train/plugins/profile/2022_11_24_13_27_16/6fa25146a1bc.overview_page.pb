�	E���\"@E���\"@!E���\"@	�~�x��?�~�x��?!�~�x��?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLE���\"@�$��I�@1E�N��%�?A&��i� �?I.�|���@Y�A�p�-�?rEagerKernelExecute 0*	�O��nɅ@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateۋh;���?!�˄J9T@)�|[�T��?1��>1`�S@:Preprocessing2F
Iterator::Model���1���?!l���p&@)G���1�?1FҶ��@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat*���P�?!<!��7?@)N{JΉ=�?1Q���]�@:Preprocessing2U
Iterator::Model::ParallelMapV2�^�S�?!�U/�@)�^�S�?1�U/�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�C�bԅ?!���QFv�?)�C�bԅ?1���QFv�?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipO�Z�7��?!2���?V@)~��g�?1f��uoX�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor4��HL�?!�?!�hC�?)4��HL�?1�?!�hC�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaps�c���?!q�*��5T@)J�i�WVj?1���b��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 25.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�56.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�~�x��?Ih�i�T@Qv�i�fp0@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�$��I�@�$��I�@!�$��I�@      ��!       "	E�N��%�?E�N��%�?!E�N��%�?*      ��!       2	&��i� �?&��i� �?!&��i� �?:	.�|���@.�|���@!.�|���@B      ��!       J	�A�p�-�?�A�p�-�?!�A�p�-�?R      ��!       Z	�A�p�-�?�A�p�-�?!�A�p�-�?b      ��!       JGPUY�~�x��?b qh�i�T@yv�i�fp0@�"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput��K�[�?!��K�[�?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterjT�[�?!�/P"��?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterd���ɫ�?!�R���?0"1
model/Conv1D_2/conv1dConv2D؝�O���?!���aV�?"1
model/Conv1D_3/conv1dConv2D�%d����?!ہ����?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput�=�"l�?!�)�/���?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad�;�QR�?!Q5z
��?"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad�yI���?!?�����?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradxͤ�#�?!��OV��?"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad7��{h�?!i��j�?Q      Y@Y&W�+�)@a�����U@q+�l(��6@y������?"�
both�Your program is POTENTIALLY input-bound because 25.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�56.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�22.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 