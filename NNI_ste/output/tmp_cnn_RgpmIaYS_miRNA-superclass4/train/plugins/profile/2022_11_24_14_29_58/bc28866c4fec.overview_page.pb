�	Y�� �(@Y�� �(@!Y�� �(@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCY�� �(@� ��"�?1��<e5�!@A�u�|�H�?I.2���?rEagerKernelExecute 0*	�"��~�g@2F
Iterator::Model��U�&�?!��E�I�H@)�e�ikD�?19�DH�@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatR�8�ߡ�?!İ:�~9@)'��2�?1�x]b�5@:Preprocessing2U
Iterator::Model::ParallelMapV2KxB�?��?!���R0@)KxB�?��?1���R0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�2��(�?!o�T� I@)φ�3���?1Ŵ�⮐!@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�n�o�>�?!��s2�2@)�n�o�>�?1��s2�2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate!���3ښ?!�R�=��+@)g���u�?1��Hc@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorT���f~?!�����@)T���f~?1�����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap=��+d�?!� ���t/@)� 3��Ol?1�qV�M�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 12.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�15.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noId���y)<@Q��ە��Q@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	� ��"�?� ��"�?!� ��"�?      ��!       "	��<e5�!@��<e5�!@!��<e5�!@*      ��!       2	�u�|�H�?�u�|�H�?!�u�|�H�?:	.2���?.2���?!.2���?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qd���y)<@y��ە��Q@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterM��$�޵?!M��$�޵?0"1
model/Conv1D_2/conv1dConv2D.\�V�p�?!�y�=�'�?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput4��)د?!���?0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad��U뱨?!�b�B.%�?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradL�,��Ҥ?!*��X���?"3
model/Conv1D_1/BiasAddBiasAdd]O㲝�?!iE���?"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose�y�6�ӡ?!Mx6p8�?"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose�W��С?!GCw�r�?"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	TransposeuM/���?!�,]}��?"-
model/Conv1D_1/ReluReluĪϹ%v�?!N"�4��?Q      Y@Y��k(�'@a����
V@q\O�7@yxc�V"�?"�
both�Your program is POTENTIALLY input-bound because 12.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�15.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�23.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 