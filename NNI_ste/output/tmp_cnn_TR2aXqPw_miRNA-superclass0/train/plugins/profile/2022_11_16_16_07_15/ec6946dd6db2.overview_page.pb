�	k��躐@k��躐@!k��躐@	�z|��@�z|��@!�z|��@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLk��躐@��)F�?1G��@A�����?I<2V��w@YNc{-��?rEagerKernelExecute 0*	t�V r@2F
Iterator::Model��N�?!�lX��Q@)�����?1�l�c��N@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat7����?!��/c�(@)����?1�ֵ\E$@:Preprocessing2U
Iterator::Model::ParallelMapV2a�hV��?!���0�� @)a�hV��?1���0�� @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�~k'JB�?!K?�O��@)�~k'JB�?1K?�O��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�!Y��?!`Mޟ�=@)�8�ߡ(�?1�����@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���hqƠ?!��0�&@)ϡU1��?1�d���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�e��a�v?!���j{�?)�e��a�v?1���j{�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��q�&�?!E�	�Y(@)�*�WY�d?1�m�}�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 23.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�38.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�z|��@I�I�	�O@QZ."3}HA@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��)F�?��)F�?!��)F�?      ��!       "	G��@G��@!G��@*      ��!       2	�����?�����?!�����?:	<2V��w@<2V��w@!<2V��w@B      ��!       J	Nc{-��?Nc{-��?!Nc{-��?R      ��!       Z	Nc{-��?Nc{-��?!Nc{-��?b      ��!       JGPUY�z|��@b q�I�	�O@yZ."3}HA@�"1
model/Conv1D_3/conv1dConv2D �}�익?! �}�익?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�׏bӡ�?!R����?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput�W�<��?!L���D�?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFiltere,�S���?!��l׫��?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput�%�-��?!lpJ�1��?0"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�[���?!��̢q�?0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad�n�?!������?"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad8%A��x�?!=�~���?"1
model/Conv1D_2/conv1dConv2D"I�?c�?!������?"C
%gradient_tape/model/Conv1D_2/ReluGradReluGrad���.ȑ�?!��5��?Q      Y@Y��u@7�)@a%D�9�U@qOZ�nA@yj���� �?"�
both�Your program is POTENTIALLY input-bound because 23.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�38.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�34.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 