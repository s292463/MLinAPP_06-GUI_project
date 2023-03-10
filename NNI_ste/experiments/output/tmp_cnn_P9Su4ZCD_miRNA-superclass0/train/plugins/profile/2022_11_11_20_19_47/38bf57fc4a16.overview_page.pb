�	-z���@-z���@!-z���@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC-z���@�T��X�?1�~�~��
@A仔�d�?I���jH�
@rEagerKernelExecute 0*	K7�A`a@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat3܀�#�?!6;�j`�<@)��6 �?1#��y�\8@:Preprocessing2F
Iterator::Models�69|ҩ?!S�bU6oB@)ղ��Hh�?1yLI�3@:Preprocessing2U
Iterator::Model::ParallelMapV2R���<�?!�Wya�M1@)R���<�?1�Wya�M1@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice0�AC��?!?֕W��-@)0�AC��?1?֕W��-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�ɨ2���?!�m�K�Q9@)��fG��?1`@Q�$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�����?!���ɐO@)K�^b,�?1���+`@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensoru�Rz��x?!PDz�s�@)u�Rz��x?1PDz�s�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��F�I�?!�r1�'�;@)nm�y��h?17)�*ݼ@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 15.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�41.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�o��M@Q��rY�D@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�T��X�?�T��X�?!�T��X�?      ��!       "	�~�~��
@�~�~��
@!�~�~��
@*      ��!       2	仔�d�?仔�d�?!仔�d�?:	���jH�
@���jH�
@!���jH�
@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�o��M@y��rY�D@�"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter*Ā%N��?!*Ā%N��?0"1
model/Conv1D_3/conv1dConv2DMLɩL��?!<�gMP�?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput��9w�?!KM��Ћ�?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterS����?!�&#;�?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput�k]:��?!�nb( �?0"1
model/Conv1D_2/conv1dConv2DĖ��%��?!��d m��?"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad"ɸ@�V�?!lSp��y�?"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad�a���?!ie�:��?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad�eI���?!�����r�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrads���?!��'�,��?Q      Y@Y�a�2�t'@a�ӭ�aV@q^���@J@y�e��q8�?"�
both�Your program is POTENTIALLY input-bound because 15.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�41.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�52.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 