�	�)@�)@!�)@	#�2Y@#�2Y@!#�2Y@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�)@i���?18��_�V�?Aط���/�?I�=~o@YCp\�M�?rEagerKernelExecute 0*	V-�q@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��J̳��?!�8��x�P@)i�k|&��?1\NovP@:Preprocessing2F
Iterator::Model�#G:#�?!����_5@)A�)V¤?1��Zh�,@:Preprocessing2U
Iterator::Model::ParallelMapV2Ֆ:����?!�.�V�@)Ֆ:����?1�.�V�@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��@�ȓ�?!���@@)�n�;2V�?1}R;�'�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�M�q��?!=��M�S@)�7�0��?1]ě[��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�����{?!0����@)�����{?10����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�!S>�?!� �j,Q@)��ǵ�bl?1<���n|�?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor��6�4De?!��$�2�?)��6�4De?1��$�2�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey=��`?!l�,'��?)y=��`?1l�,'��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 22.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�51.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9#�2Y@I�|Z؄R@Q���}a7@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	i���?i���?!i���?      ��!       "	8��_�V�?8��_�V�?!8��_�V�?*      ��!       2	ط���/�?ط���/�?!ط���/�?:	�=~o@�=~o@!�=~o@B      ��!       J	Cp\�M�?Cp\�M�?!Cp\�M�?R      ��!       Z	Cp\�M�?Cp\�M�?!Cp\�M�?b      ��!       JGPUY#�2Y@b q�|Z؄R@y���}a7@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterh�}�s:�?!h�}�s:�?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput�����?!n!h6�?0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad�r�H)�?!�]�K�%�?"1
model/Conv1D_2/conv1dConv2D�L�P�ơ?!��_(��?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad�ۤy<��?!(,~w��?"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits� t�?!�d���?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�'�q��?!��0�4�?0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradhX$hIL�?!��%��)�?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�`� �?!3�����?0"C
%gradient_tape/model/Conv1D_2/ReluGradReluGrad�hY*��?!��h�q�?Q      Y@Y<Eg@()@aY����U@qU�pn)5C@y���8���?"�
both�Your program is POTENTIALLY input-bound because 22.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�51.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�38.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 