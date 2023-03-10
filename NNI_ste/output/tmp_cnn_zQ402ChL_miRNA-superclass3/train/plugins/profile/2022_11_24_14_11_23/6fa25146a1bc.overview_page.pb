�	��Os�+@��Os�+@!��Os�+@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC��Os�+@�Yh�4� @11�߄B�?A,���o�?I����@rEagerKernelExecute 0*	�K7�A�e@2F
Iterator::Model��R�?!]S~�F@)1��*��?1�C<G@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat-Ӿ��?!�6��:@)�w(
�?1h,��`�6@:Preprocessing2U
Iterator::Model::ParallelMapV2��r���?!�d\�}�*@)��r���?1�d\�}�*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���7�?!�z�q1@)7��:r��?1��j��%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip稣�j�?!�JK@)���P1Ώ?1&�^��!@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�@j'�?!��YH3�@)�@j'�?1��YH3�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�q4GV~y?!�Ag��}@)�q4GV~y?1�Ag��}@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap |(�?!+��-3@)��@���h?1:юw��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 60.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�30.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI:7?:�V@Q3F,"@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�Yh�4� @�Yh�4� @!�Yh�4� @      ��!       "	1�߄B�?1�߄B�?!1�߄B�?*      ��!       2	,���o�?,���o�?!,���o�?:	����@����@!����@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q:7?:�V@y3F,"@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter$k�����?!$k�����?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilternR~�ɦ?!�ޗ�G)�?0"1
model/Conv1D_2/conv1dConv2Dg8L�0��?!~�^���?"1
model/Conv1D_3/conv1dConv2Dė%���?!oc�{a��?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad(zXFc�?!���3�?"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad���֋��?!����y�?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput�K�ܢ?!׼L-���?0"1
model/Conv1D_1/conv1dConv2D�ȭ���?!�u��d��?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad�����?!���ܖ�?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput�.�Ѡ?!5:t�+�?0Q      Y@Y@n]�G*@a8R4��U@q��P�a�L@yA&��6�?"�
both�Your program is POTENTIALLY input-bound because 60.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�30.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�57.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 