�	�*��p-@�*��p-@!�*��p-@	7��Ih�?7��Ih�?!7��Ih�?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�*��p-@�T3k) �?1�G7¢�@AAc&Q/�?I���ڧ%@Y��2p@�?rEagerKernelExecute 0*	�O��n�d@2F
Iterator::Model��>+�?!�\��lcJ@)fٓ���?1��]1.C@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�1^�?!�t�{��6@)�Q�y9�?1��,��g2@:Preprocessing2U
Iterator::Model::ParallelMapV2�$��}8�?!c3"H��,@)�$��}8�?1c3"H��,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����4)�?!�Y�0)@)����4)�?1�Y�0)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate}$%=��?!֡�qB2@)S[� ��?1DL��%�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipW@�ճ?!�+��G@)��#0�?1n��ÄE@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����{?!��W�?�@)����{?1��W�?�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���:��?!j��s��3@)ĕ�wF[e?1H��  l�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 9.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�72.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no97��Ih�?I�i:₩T@Q��Z���.@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�T3k) �?�T3k) �?!�T3k) �?      ��!       "	�G7¢�@�G7¢�@!�G7¢�@*      ��!       2	Ac&Q/�?Ac&Q/�?!Ac&Q/�?:	���ڧ%@���ڧ%@!���ڧ%@B      ��!       J	��2p@�?��2p@�?!��2p@�?R      ��!       Z	��2p@�?��2p@�?!��2p@�?b      ��!       JGPUY7��Ih�?b q�i:₩T@y��Z���.@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�R�㫻�?!�R�㫻�?0"1
model/Conv1D_2/conv1dConv2DCܮ�ek�?!h�ֈ��?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad�U�c�E�?!謉��$�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad#��Ǯ�?!�贬���?"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	TransposeWy/\��?!<�I\>�?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilterN�D�ے�?!&5r�^��?0"3
model/Conv1D_1/BiasAddBiasAdd�g��Ģ?! ��%��?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter/�����?!(��^�?0"-
model/Conv1D_1/ReluRelu�wo�?!�
�@��?"}
^gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer	Transpose����ᥡ?!�������?Q      Y@Ym۶m۶)@a�$I�$�U@q�L�K�@@y~��%��?"�
both�Your program is POTENTIALLY input-bound because 9.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�72.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�33.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 