� 	���y�#@���y�#@!���y�#@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC���y�#@N^d~� @1(+����?A�B����?I`��F@rEagerKernelExecute 0*	"��~jٕ@2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Mapl#����?!P�p=� T@)����y�?1��z
�S@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapW��:��?!;`����*@)�-W?6��?1����&@:Preprocessing2F
Iterator::ModelR~R���?!��A�@)�`6��?1�+�@��
@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat��<L�?!�}����?)YvQ���?1w�Ċ�?:Preprocessing2U
Iterator::Model::ParallelMapV2~7ݲC��?!^3��T�?)~7ݲC��?1^3��T�?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���n��?!�k?���?)��rf�B�?1M��%w�?:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat����Ǔ?!`�9A�?)F�j���?1�R����?:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::PrefetchY�8��m�?!t����?)Y�8��m�?1t����?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip}�X��L�?!����Z�.@)�($��;|?1���ZF��?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorS%��R�w?!�ۈ��?)S%��R�w?1�ۈ��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice$c���q?!�	�k��?)$c���q?1�	�k��?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range�r�9>Zl?!��0�@��?)�r�9>Zl?1��0�@��?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�2�FY�y?!���A��?)1?74e�_?1P���N��?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor���O?!<3(б?)���O?1<3(б?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 21.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�63.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI=�BN#uU@Q���V,@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	N^d~� @N^d~� @!N^d~� @      ��!       "	(+����?(+����?!(+����?*      ��!       2	�B����?�B����?!�B����?:	`��F@`��F@!`��F@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q=�BN#uU@y���V,@�"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput4elf�?!4elf�?0"1
model/Conv1D_3/conv1dConv2Dc�����?!`�ׂ=��?"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad˄�]�/�?!�l;�9i�?"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits���AU�?!��.M�>�?"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInputJוZ���?!�!�1���?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��U2�?!�.�.��?0"C
%gradient_tape/model/Conv1D_2/ReluGradReluGrad����̞?!�`=����?"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter���wۚ?!����p�?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�#��?!���&��?0"1
model/Conv1D_2/conv1dConv2D�-Y��?!b�p��?Q      Y@Y��)kʚ2@a֔5eMYT@q���V��-@ypK��v��?"�
both�Your program is POTENTIALLY input-bound because 21.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�63.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�14.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 