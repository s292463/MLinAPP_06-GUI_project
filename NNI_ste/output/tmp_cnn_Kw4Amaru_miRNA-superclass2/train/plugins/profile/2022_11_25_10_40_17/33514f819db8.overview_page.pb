�	�y���+@�y���+@!�y���+@	�8@$�@�8@$�@!�8@$�@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�y���+@5�+-#��?1�ӹ��P@AM�O�t?I2��p@Y)%����?rEagerKernelExecute 0*	R���uc@2F
Iterator::Modelb1�Z{��?!�D��\G@)K�|%�?1W�c�J>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�_>Y1\�?!	'� �:@)����?1��WhM�6@:Preprocessing2U
Iterator::Model::ParallelMapV2��I�2�?!�L�"o0@)��I�2�?1�L�"o0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��};�?!�<�J@)��]/M�?1 |VI($@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice*�"��?!��j�F�@)*�"��?1��j�F�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateR,���?!7~iAl�,@)y:W���?1�h���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�M�#~�z?!nrb��@)�M�#~�z?1nrb��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapY�e0F$�?!�E��e0@)<hv�[�h?1���:��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 4.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�54.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�8@$�@I@{fxL@Q��v��.C@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	5�+-#��?5�+-#��?!5�+-#��?      ��!       "	�ӹ��P@�ӹ��P@!�ӹ��P@*      ��!       2	M�O�t?M�O�t?!M�O�t?:	2��p@2��p@!2��p@B      ��!       J	)%����?)%����?!)%����?R      ��!       Z	)%����?)%����?!)%����?b      ��!       JGPUY�8@$�@b q@{fxL@y��v��.C@�"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFiltera
�s��?!a
�s��?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�^ϝF��?!�L�s<��?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFiltera;	&�q�?!It��n-�?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?�s���?!1��Fb��?0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGradp�N�n�?!(�+�P:�?"1
model/Conv1D_2/conv1dConv2D�Yo֗?!AL!�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad�wN#{�?!��#c/�?"1
model/Conv1D_3/conv1dConv2D#{
�J�?!ø���G�?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad��׆��?!�u���?"C
%gradient_tape/model/Conv1D_2/ReluGradReluGrad8�w�<��?!�1��R��?Q      Y@YAd�W�,)@ax��g�U@q�hjq];@yk����?"�
device�Your program is NOT input-bound because only 4.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�54.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�27.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 