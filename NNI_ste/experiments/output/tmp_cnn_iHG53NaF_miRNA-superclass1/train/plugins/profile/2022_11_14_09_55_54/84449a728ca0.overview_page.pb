�	ҌE���@ҌE���@!ҌE���@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCҌE���@��\����?1�BB5@Ag��)�?IW횐��@rEagerKernelExecute 0*	�"��~�e@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�6��nf�?!���.	@@)X�<���?1L%�{<@:Preprocessing2U
Iterator::Model::ParallelMapV2�e���~�?!���6@)�e���~�?1���6@:Preprocessing2F
Iterator::Model�6�[ �?!e��+WC@)�\�	��?1Ș��q�0@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceTV��Dו?!�����(@)TV��Dו?1�����(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateY�U���?!Ci�
/7@)^f�(�7�?1�d�3��%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip3��J&�?!��ԨN@)�����?1�vU@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorO#-��#|?!(���@)O#-��#|?1(���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�%䃞�?!����j8@)��L�na?1ėܘ���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 15.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�48.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIa8��P@Q>�3���A@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��\����?��\����?!��\����?      ��!       "	�BB5@�BB5@!�BB5@*      ��!       2	g��)�?g��)�?!g��)�?:	W횐��@W횐��@!W횐��@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qa8��P@y>�3���A@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��CuJ-�?!��CuJ-�?0"1
model/Conv1D_2/conv1dConv2Dܑw{�r�?!2�]�#P�?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput�9�m�?!��z��?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�\Y��?!ro�T�a�?0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrade�ł�c�?!%%y��?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrado3:���?!�khx�*�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad+>�,��?!Rs>�g�?"1
model/Conv1D_3/conv1dConv2D�6�S՟?!��De�?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�����?!'�r�E�?0"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transposeto�OIu�?!V�S�?Q      Y@Y$I�$I�,@a۶m۶mU@q��͡ɹE@y�P+��f�?"�
both�Your program is POTENTIALLY input-bound because 15.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�48.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�43.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 