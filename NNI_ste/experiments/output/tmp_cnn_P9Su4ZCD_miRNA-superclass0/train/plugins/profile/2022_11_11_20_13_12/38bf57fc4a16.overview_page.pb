� 	&��|t"@&��|t"@!&��|t"@      ��!       "{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:&��|t"@���H��?1ß���@Ie�`TR'	@rEagerKernelExecute 0*	q=
ףV�@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap@��wԘ�?!�6t_�M@)`tys�V�?1cY��kJ@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map���ʅ�?!���z�<@)��Y,�?1c���>7@:Preprocessing2F
Iterator::Modelʈ@�t�?!7�Ӆ�!@)�J�����?1�r��m�@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatyx��e�?!���v@):��l�?1�g)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate"ߥ�%�?!\@���@)��8�Z�?1��Ǣ�g@:Preprocessing2U
Iterator::Model::ParallelMapV2ލ�A��?!��'f;�@)ލ�A��?1��'f;�@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�9�����?!������@)R�d=��?1Z�|��?:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch�!�{��?!,�Th��?)�!�{��?1,�Th��?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZiptB�K8�?!�
�!QO@)�eM,�}?1�Ǟ
-�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceM�St$w?!��b��6�?)M�St$w?1��b��6�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���V_]u?!L���]n�?)���V_]u?1L���]n�?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range��-�r?!�p�{���?)��-�r?1�p�{���?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate��s�f|?!c�k}�?)���|	U?1Ӈb8�?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensorC�8
Q?!�YO��?)C�8
Q?1�YO��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 10.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�34.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIv�PAF@Q�����K@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���H��?���H��?!���H��?      ��!       "	ß���@ß���@!ß���@*      ��!       2      ��!       :	e�`TR'	@e�`TR'	@!e�`TR'	@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qv�PAF@y�����K@�"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter���2D��?!���2D��?0"1
model/Conv1D_3/conv1dConv2D@#���k�?!g\���3�?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputb,ݗ�G�?!����t��?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilternB��?!C�Z<��?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput%��� ?!�����?0"1
model/Conv1D_2/conv1dConv2D���}��?!�8�s���?"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad�Ce����?!"�_���?"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad�q�����?!;��&/�?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad/��NJ�?!�kˣ�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad�� �Ya�?!���	�?Q      Y@Y/�袋.2@au�E]tT@q0���0@"�
both�Your program is POTENTIALLY input-bound because 10.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�34.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�16.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 