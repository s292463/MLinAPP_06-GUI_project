�	�WXp/5@�WXp/5@!�WXp/5@	�I	��-@�I	��-@!�I	��-@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�WXp/5@P��n�?1f����l/@A���
a5v?I��p�Q%�?Y��"ڎi	@rEagerKernelExecute 0*	V-�f@2F
Iterator::ModelJ�>�ɳ?!-�Z��'E@)XY����?1��� �`<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�����?!L��T<@)�~�f+�?1�F#��8@:Preprocessing2U
Iterator::Model::ParallelMapV2w�x��?!e�σ��+@)w�x��?1e�σ��+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�UIdd�?!��/%+@)�UIdd�?1��/%+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate9�Z�̤?!=��<6@)��.Q�5�?1�$s�aT!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�s)�*��?!�E�NG�L@)9�M�a��?1~y=M�Z@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��B�ʠz?!*�l�w@)��B�ʠz?1*�l�w@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��_�|x�?!0�@'�8@)�9��j?1<���q��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 15.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.moderate"�8.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�I	��-@I��°H�%@Q�跊R@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	P��n�?P��n�?!P��n�?      ��!       "	f����l/@f����l/@!f����l/@*      ��!       2	���
a5v?���
a5v?!���
a5v?:	��p�Q%�?��p�Q%�?!��p�Q%�?B      ��!       J	��"ڎi	@��"ڎi	@!��"ڎi	@R      ��!       Z	��"ڎi	@��"ڎi	@!��"ڎi	@b      ��!       JGPUY�I	��-@b q��°H�%@y�跊R@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter@�ց\�?!@�ց\�?0"1
model/Conv1D_2/conv1dConv2DQCB'��?!�( ����?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput��q��0�?!Ŏ\M��?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFiltera�6j�A�?!��S8�?0"1
model/Conv1D_3/conv1dConv2Dk�����?!���֩�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradX���\�?!g�Tl"&�?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradc!���~�?!��� 6�?"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose�t��Tʝ?!ܥ8��?"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transposeh�Qy��?!bðO��?"3
model/Conv1D_1/BiasAddBiasAdd�S4�I�?!�4����?Q      Y@Y@n]�G*@a8R4��U@q��=�G�@y*����?"�
both�Your program is MODERATELY input-bound because 15.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�8.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 