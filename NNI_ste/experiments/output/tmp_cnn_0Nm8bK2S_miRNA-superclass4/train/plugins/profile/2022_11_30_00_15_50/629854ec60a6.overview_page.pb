�	DL�$z�!@DL�$z�!@!DL�$z�!@	��+]@��+]@!��+]@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCDL�$z�!@�y0H��?1V���n?@I��X� @YR%�S;�?rEagerKernelExecute 0*	�/�$rg@2F
Iterator::Model�O����?!u{' !�F@){JΉ=��?1��2Բ�@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatX���ާ�?!�G�Ȩ�;@)^�c@�z�?1�l��.s8@:Preprocessing2U
Iterator::Model::ParallelMapV2z�ؘ��?!�*ӯ�(@)z�ؘ��?1�*ӯ�(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�gyܝ�?!L��`�&@)�gyܝ�?1L��`�&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��g�e�?!�����|K@)Z���֑?1k�'e�"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���u�?!���7a0@)J|����?1e����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�w��Dgy?!n�m�s
@)�w��Dgy?1n�m�s
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��!6X8�?!�9,�b�1@)I��Z��g?1������?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 4.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�23.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��+]@Iȑ�BuU<@Q�<o�Q@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�y0H��?�y0H��?!�y0H��?      ��!       "	V���n?@V���n?@!V���n?@*      ��!       2      ��!       :	��X� @��X� @!��X� @B      ��!       J	R%�S;�?R%�S;�?!R%�S;�?R      ��!       Z	R%�S;�?R%�S;�?!R%�S;�?b      ��!       JGPUY��+]@b qȑ�BuU<@y�<o�Q@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�<3�k�?!�<3�k�?0"1
model/Conv1D_2/conv1dConv2D�@$��N�?!��+���?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradW��6bc�?!D;����?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradWWJ��j�?!Ѹ6��?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput��F�^G�?!�C�m2�?0"1
model/Conv1D_3/conv1dConv2D���-Vأ?!���3=��?"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transposen�L��?!`�?�~��?"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose4�!K��?!�jrAh��?"3
model/Conv1D_1/BiasAddBiasAddv�z���?!���p;*�?"-
model/Conv1D_1/ReluRelu#.�({�?!T�)��9�?Q      Y@Y��u@7�)@a%D�9�U@q7�'N8J@y�!H�y��?"�
both�Your program is POTENTIALLY input-bound because 4.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�23.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�52.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 