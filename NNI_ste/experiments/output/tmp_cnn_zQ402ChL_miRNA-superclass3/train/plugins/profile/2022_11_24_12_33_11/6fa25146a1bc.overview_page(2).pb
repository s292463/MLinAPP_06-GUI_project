�	�R���p@�R���p@!�R���p@	'[6���?'[6���?!'[6���?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�R���p@����B-�?1���X�m@A���b)��?I�����v>@Y�\n0�a�?rEagerKernelExecute 0*	&��Cot@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate}��F�?!C�J�|Q@)=E7��?1���R�Q@:Preprocessing2F
Iterator::Model'�5��?!9%��5�5@)2�3/�ݧ?1�-�L�,@:Preprocessing2U
Iterator::Model::ParallelMapV27�n�e��?!\9<.?N@)7�n�e��?1\9<.?N@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��$W@�?!ub�`�@)-?p�'�?1��l\
@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��6�h�?!��݊��S@)��l#~?1��5%� @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensoreȱ��x?!����[��?)eȱ��x?1����[��?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��+,��?!L�����Q@)�(�1kl?1䌓����?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor�W�\i?!,��/M�?)�W�\i?1,��/M�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice���Q��\?!Q�ELF�?)���Q��\?1Q�ELF�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�11.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9'[6���?I�AnT
�'@Q�}�$V@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����B-�?����B-�?!����B-�?      ��!       "	���X�m@���X�m@!���X�m@*      ��!       2	���b)��?���b)��?!���b)��?:	�����v>@�����v>@!�����v>@B      ��!       J	�\n0�a�?�\n0�a�?!�\n0�a�?R      ��!       Z	�\n0�a�?�\n0�a�?!�\n0�a�?b      ��!       JGPUY'[6���?b q�AnT
�'@y�}�$V@�"1
model/Conv1D_2/conv1dConv2Dlu�	Q��?!lu�	Q��?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter���F��?!�I��KK�?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput�=D#���?!Yne�G�?0"1
model/Conv1D_3/conv1dConv2D��Z�A8�?!q����?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput3C�ú��?!�u�/j��?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�#��U�?!�������?0"1
model/Conv1D_4/conv1dConv2D��eI��?!v���?"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput��4����?!��S�J�?0"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�w�wJ��?!�n���?0"1
model/Conv1D_1/conv1dConv2D�b�'[h�?!�O�7e�?Q      Y@Y���"�@a�!Tҍ�W@q�.��1�5@yʂ	e�{Q?"�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�21.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 