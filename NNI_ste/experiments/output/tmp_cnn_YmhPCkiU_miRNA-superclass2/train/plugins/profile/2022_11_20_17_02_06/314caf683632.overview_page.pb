�	L�4��1@L�4��1@!L�4��1@	(e� <@(e� <@!(e� <@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLL�4��1@��ڊ���?1N�����@A�G�Ȱ��?I	���7@Y�E`�o`�?rEagerKernelExecute 0*	��Q��r@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenatenR�X�;�?!���P>P@)���zi��?1D��C�O@:Preprocessing2F
Iterator::Model*:��H�?!�D�yH�8@)EJ�y�?1_��H6e1@:Preprocessing2U
Iterator::Model::ParallelMapV2T�:��?!� �H�@)T�:��?1� �H�@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�d�?!Ɗ�r�w@)����>�?1�i�0@@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�Ȱ�72�?!߮��-�R@)1`�U,~�?1��$b	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorM�~2Ƈy?!�A�o @)M�~2Ƈy?1�A�o @:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor=�- �n?!�
���?)=�- �n?1�
���?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�o�4(��?!�M��zP@)�MG 7�g?1�[zO�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicee����`[?!���C���?)e����`[?1���C���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 19.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�34.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9(e� <@I��X��#K@QB��D@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��ڊ���?��ڊ���?!��ڊ���?      ��!       "	N�����@N�����@!N�����@*      ��!       2	�G�Ȱ��?�G�Ȱ��?!�G�Ȱ��?:		���7@	���7@!	���7@B      ��!       J	�E`�o`�?�E`�o`�?!�E`�o`�?R      ��!       Z	�E`�o`�?�E`�o`�?!�E`�o`�?b      ��!       JGPUY(e� <@b q��X��#K@yB��D@�"1
model/Conv1D_2/conv1dConv2D]��e�?!]��e�?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�fil̰?!�5�J�?0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad�)T��?!�?4X�?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad��\2��?!V}�d}�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad�|��/��?!;n�Z�0�?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput����?!������?0"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose��� v�?!cI'�*��?"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose�؇,�ܟ?!�����?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter_��{~��?!V�i�~�?0"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose�����?!���s�m�?Q      Y@Yݘ��V+@a`�.�U@qM�B�8@y�	�
���?"�
both�Your program is POTENTIALLY input-bound because 19.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�34.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�24.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 