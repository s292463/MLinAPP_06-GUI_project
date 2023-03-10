�	�&"��v@�&"��v@!�&"��v@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC�&"��v@�+f�l@1Z�$�9�\@A�=�WX�?ID1y�$5@rEagerKernelExecute 0*	"��~j0c@2F
Iterator::Model֫��$�?!I윌�I@)1�:9Cq�?1�@S�=@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateE)!XU/�?!J/C�q=@)t`9B�?1�Or��:@:Preprocessing2U
Iterator::Model::ParallelMapV2{�V��נ?!�{��m5@){�V��נ?1�{��m5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�����Q�?!a-�k��(@)�7�-:�?1���S @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��(�?!��cs_H@)�+�S�?1p+�l�}@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora�$��z?!pe�my@)a�$��z?1pe�my@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor�ڧ�1e?!�G�\��?)�ڧ�1e?1�G�\��?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapS��F;n�?!���t,?@)� ��^�c?1r)K�[�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice�d��~�]?!H�\���?)�d��~�]?1H�\���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 62.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�5.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI$a��#Q@Qr�{�Yp?@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�+f�l@�+f�l@!�+f�l@      ��!       "	Z�$�9�\@Z�$�9�\@!Z�$�9�\@*      ��!       2	�=�WX�?�=�WX�?!�=�WX�?:	D1y�$5@D1y�$5@!D1y�$5@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q$a��#Q@yr�{�Yp?@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�����(�?!�����(�?0"1
model/Conv1D_2/conv1dConv2D��S���?!�?(���?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput>�4��e�?!e����?0"1
model/Conv1D_3/conv1dConv2D
���?!6ٻ����?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter_jյs�?!�)gAI��?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput|	M:�'�?!-�9��y�?0"1
model/Conv1D_4/conv1dConv2Duu�C�_�?!�MW��?"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput�6��t�?!�V�=W��?0"1
model/Conv1D_1/conv1dConv2Do��ݥ �?!�H��
�?"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter8��ES�?!
G,�&t�?0Q      Y@Y�{�1m@aD�,��W@q�%��9�@@y_L�u��S?"�
both�Your program is POTENTIALLY input-bound because 62.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�5.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�33.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 