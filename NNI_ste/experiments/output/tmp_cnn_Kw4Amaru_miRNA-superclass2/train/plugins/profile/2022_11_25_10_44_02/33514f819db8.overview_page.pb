�	�*��@�*��@!�*��@	WS�Y@WS�Y@!WS�Y@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�*��@?o*Ral�?1,*�t��@A�>s֧�?Ii��U[@Y�"��]��?rEagerKernelExecute 0*	̡E���e@2F
Iterator::Model�	/���?!����UF@)�"��Jv�?1SAЯ?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�$���?![�eo@@)l�V^�?�?1�\�L<@:Preprocessing2U
Iterator::Model::ParallelMapV2���DR�?!�66hv�)@)���DR�?1�66hv�)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice*kg{�?!qa�Vx�&@)*kg{�?1qa�Vx�&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�c�ZB�?!_����0@)�
��捃?1�f�"�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�x[�ٸ?!4�QZz�K@)o+�6+�?1
���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��<e5}?!��^�HB@)��<e5}?1��^�HB@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap����?!�``��`2@)�"�~�f?1iI�(��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 7.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�43.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9WS�Y@I\�����I@Q��5�-F@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	?o*Ral�??o*Ral�?!?o*Ral�?      ��!       "	,*�t��@,*�t��@!,*�t��@*      ��!       2	�>s֧�?�>s֧�?!�>s֧�?:	i��U[@i��U[@!i��U[@B      ��!       J	�"��]��?�"��]��?!�"��]��?R      ��!       Z	�"��]��?�"��]��?!�"��]��?b      ��!       JGPUYWS�Y@b q\�����I@y��5�-F@�"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput��Ȧ��?!��Ȧ��?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��!\�?!�`Cd���?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter=3w�	�?!�-��| �?0"1
model/Conv1D_2/conv1dConv2D��GT���?!>�+��?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput8Kc\���?!��g%�?0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad��N_ٛ?!�o����?"1
model/Conv1D_3/conv1dConv2D����2j�?!z��ܠ��?"C
%gradient_tape/model/Conv1D_2/ReluGradReluGrad�=o�=d�?!Z���?�?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter9	�v���?!�O���?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrade��Y�c�?!y���u�?Q      Y@Y�=����(@aFX�i��U@q�F��a'7@yRٕ@�x�?"�
both�Your program is POTENTIALLY input-bound because 7.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�43.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�23.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 