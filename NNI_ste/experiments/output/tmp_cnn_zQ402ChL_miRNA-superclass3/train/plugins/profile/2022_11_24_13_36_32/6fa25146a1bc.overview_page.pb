�	�k��v@�k��v@!�k��v@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC�k��v@UD� �?1�E�n4�s@AS@�� k�?I����F@rEagerKernelExecute 0*	ףp=�d@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate\Ɏ�@��?!�O*`
=B@)4J��%��?10�b��A@:Preprocessing2F
Iterator::Model���h o�?!�Z�o��D@)bX9�Ȧ?1��EH
;@:Preprocessing2U
Iterator::Model::ParallelMapV2��0�*�?!��4I�,@)��0�*�?1��4I�,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��B��?!��IOM@)J�U��?1ME��e#@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�K�K�1�?!YE��ޗ%@)��{�q�?1ON�U�2@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�[���u?!�xLl^�	@)�[���u?1�xLl^�	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap:w�^�"�?!^©g8&C@)~Q��B�h?1)T���%�?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor]�����a?!O�"+�1�?)]�����a?1O�"+�1�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceF�̱��^?!IEʍ=3�?)F�̱��^?1IEʍ=3�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�12.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI ;���)@Q�����U@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	UD� �?UD� �?!UD� �?      ��!       "	�E�n4�s@�E�n4�s@!�E�n4�s@*      ��!       2	S@�� k�?S@�� k�?!S@�� k�?:	����F@����F@!����F@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q ;���)@y�����U@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��'�{��?!��'�{��?0"1
model/Conv1D_2/conv1dConv2Df�� ��?!*.m!�5�?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput�IE�e�?!4D�h�?0"1
model/Conv1D_3/conv1dConv2D,��y��?!���I��?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�j@��?!�d���e�?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput�D�ڎ�?!��ƷP2�?0"1
model/Conv1D_4/conv1dConv2D�`pM'�?!�Y2���?"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput2u}�?!������?0"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter`�+s�ȏ?!v����?0"1
model/Conv1D_1/conv1dConv2D�3�.έ�?!EћMwo�?Q      Y@Y����@a���!5�W@q�����G@yh��"��U?"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�12.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�47.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 