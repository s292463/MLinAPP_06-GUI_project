�	�o��1=@�o��1=@!�o��1=@	Ͳ^k3@Ͳ^k3@!Ͳ^k3@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�o��1=@�C6�.��?1KXc'��?Ae��)1�?I�j�	@Y�GT�n�?rEagerKernelExecute 0*	���S��d@2F
Iterator::Model-��;���?!��5&��E@)hX���ާ?1��<!X<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��8�#�?!�VӃ�?@)��RAEէ?1-^#&<@:Preprocessing2U
Iterator::Model::ParallelMapV2��yS��?!�%^Vx?/@)��yS��?1�%^Vx?/@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���>e�?!�-`��'@)���>e�?1�-`��'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�pz�?!W��u&L@)����qn�?1k@R��@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate|��l;m�?!���'N1@)��g\8�?18bc��>@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorK�H��rz?!/��ie@)K�H��rz?1/��ie@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�3���?!�K��2@)b��!��b?1K�����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 25.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�50.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9~Ͳ^k3@I?�wKS@Q�$Բ�3@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�C6�.��?�C6�.��?!�C6�.��?      ��!       "	KXc'��?KXc'��?!KXc'��?*      ��!       2	e��)1�?e��)1�?!e��)1�?:	�j�	@�j�	@!�j�	@B      ��!       J	�GT�n�?�GT�n�?!�GT�n�?R      ��!       Z	�GT�n�?�GT�n�?!�GT�n�?b      ��!       JGPUY~Ͳ^k3@b q?�wKS@y�$Բ�3@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�|����?!�|����?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput�c���?!Ip�іȵ?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�PXf�O�?!�]�?0"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad��{�h�?!��i�U��?"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter2�|�i�?!4��l�?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputn��T��?!�k�U|��?0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad�8�ϋˠ?!��ܤ���?"1
model/Conv1D_2/conv1dConv2DGh�n}�?!�	{]��?"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput����!�?!�ӷ|��?0"1
model/Conv1D_3/conv1dConv2Dǻ�����?!p��Pؖ�?Q      Y@Y@n]�G*@a8R4��U@qٓ ���A@y������?"�
both�Your program is POTENTIALLY input-bound because 25.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�50.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�35.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 