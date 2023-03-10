�	��
~�@��
~�@!��
~�@	�Z1���@�Z1���@!�Z1���@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��
~�@�w��m�?1,~SX� �?AhZbe4�?I/��dƛ@Yʇ�j�j�?rEagerKernelExecute 0*	ʡE��e@2F
Iterator::ModelT�:��?!�;ZFK
J@)YR�>�G�?1)�bB	�>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��/Ie��?!/�#2��<@)A��_��?1��4z��8@:Preprocessing2U
Iterator::Model::ParallelMapV2Pr�Md�?!��QJ�P5@)Pr�Md�?1��QJ�P5@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipW!�'�>�?!ĥ���G@)���Y��?1C�v���@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��W;�s�?!�U�p@)��W;�s�?1�U�p@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���m�?!
�1Ke�%@)�s�Lh�?1�Y�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��	L�u{?!Orv���@)��	L�u{?1Orv���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�>;�b�?!�#���>)@)>u�Rz�g?1�����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 21.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�53.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�Z1���@Ik��z�R@Ql(\5�t4@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�w��m�?�w��m�?!�w��m�?      ��!       "	,~SX� �?,~SX� �?!,~SX� �?*      ��!       2	hZbe4�?hZbe4�?!hZbe4�?:	/��dƛ@/��dƛ@!/��dƛ@B      ��!       J	ʇ�j�j�?ʇ�j�j�?!ʇ�j�j�?R      ��!       Z	ʇ�j�j�?ʇ�j�j�?!ʇ�j�j�?b      ��!       JGPUY�Z1���@b qk��z�R@yl(\5�t4@�"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��wY�?!��wY�?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�ѳ���?!�qp,,�?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput�-��?�?!�������?0"1
model/Conv1D_3/conv1dConv2Dc�_o�Q�?!՗��o��?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput#:n|�k�?!_)����?0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad�vJ�z�?!�>)���?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�C
^��?!��/�`�?0"1
model/Conv1D_2/conv1dConv2D�3Z'��?!��_R"x�?"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput=-��ƍ�?!e8}��<�?0"�
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits~G��%�?!�B춱��?Q      Y@Ym۶m۶)@a�$I�$�U@q�sK��@@y��f���?"�
both�Your program is POTENTIALLY input-bound because 21.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�53.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�33.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 