�	��8Q @��8Q @!��8Q @	.���f@.���f@!.���f@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��8Q @|��8G�?13p@KW@A�h㈵�T?I��Dg�%@Y��ʡE�?rEagerKernelExecute 0*	m��ʫt@2U
Iterator::Model::ParallelMapV2b��U��?!NB����K@)b��U��?1NB����K@:Preprocessing2F
Iterator::Model����/�?!$� �jR@)�-�l�I�?1��:���1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�f���§?!&kj,@)��X��?1�U��M�'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice<�$��?!���NnP@)<�$��?1���NnP@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateH¾�D��?!���Ü�"@)VW@܅?1��r��	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorOt	�~?!��6�@)Ot	�~?1��6�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip*��K�?!o���?U:@)W�}W�{?1倳�c� @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapհ��T�?!�C�Tx$@)�P�,i?1�ڕ�}��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 5.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�28.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9/���f@Ibz�4�	A@Q܌�n�/O@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	|��8G�?|��8G�?!|��8G�?      ��!       "	3p@KW@3p@KW@!3p@KW@*      ��!       2	�h㈵�T?�h㈵�T?!�h㈵�T?:	��Dg�%@��Dg�%@!��Dg�%@B      ��!       J	��ʡE�?��ʡE�?!��ʡE�?R      ��!       Z	��ʡE�?��ʡE�?!��ʡE�?b      ��!       JGPUY/���f@b qbz�4�	A@y܌�n�/O@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��s8���?!��s8���?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput��L���?!�@BB��?0"1
model/Conv1D_3/conv1dConv2D��@�X©?!"g�g-��?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterZ݂v��?!xs�r��?0"1
model/Conv1D_2/conv1dConv2D����?!󌣊9�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad� ���ڡ?!!�w�t�?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad���(���?!я
�_�?"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose�n�c�қ?!��I��?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFiltero�rқ?!B(�?��?0"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose�j�?!q�� ���?Q      Y@Y��u@7�)@a%D�9�U@q�<y	0=@y7#vݰ?"�
both�Your program is POTENTIALLY input-bound because 5.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�28.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�29.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 