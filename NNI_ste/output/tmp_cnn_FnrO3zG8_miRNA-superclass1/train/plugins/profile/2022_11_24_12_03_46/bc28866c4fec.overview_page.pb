�!	��n��@��n��@!��n��@	��?j�@��?j�@!��?j�@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��n��@�O�eo�?1��U�@��?A��u�ӥ?ID�|�f@Y�x�Z�?rEagerKernelExecute 0*	&���	�@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapݴ�!*�?!)��[{fG@)�c�±�?1ؤz���E@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Mapw�Nyt#�?!�1���5F@)[�[!�F�?1e��<]D@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat�a0�̭?!J�J�RJ@)�$@M-[�?1&��5��@:Preprocessing2F
Iterator::Model& ��*Q�?!nC����@)�d�F ^�?1Š��@:Preprocessing2U
Iterator::Model::ParallelMapV2��6�4D�?!�]ҭ@)��6�4D�?1�]ҭ@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatet���מ�?!׎��Q��?)�t����?1�h����?:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��b)���?!��H~U�?)��}�?1y+A-���?:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch���R���?!���ʭ�?)���R���?1���ʭ�?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�J��q(�?!�E�^��H@)%���4�?1�9����?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice�r��+|?!�W�X�?)�r��+|?1�W�X�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor9Q�?�{?!�9P���?)9Q�?�{?1�9P���?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range�聏��s?!hCt5���?)�聏��s?1hCt5���?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate>��j�#�?!г����?)*��g\8`?1�KPB���?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor��u6�Y?!�_R.���?)��u6�Y?1�_R.���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 6.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�43.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t22.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9��?j�@I����P@Q�\�_�;@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�O�eo�?�O�eo�?!�O�eo�?      ��!       "	��U�@��?��U�@��?!��U�@��?*      ��!       2	��u�ӥ?��u�ӥ?!��u�ӥ?:	D�|�f@D�|�f@!D�|�f@B      ��!       J	�x�Z�?�x�Z�?!�x�Z�?R      ��!       Z	�x�Z�?�x�Z�?!�x�Z�?b      ��!       JGPUY��?j�@b q����P@y�\�_�;@�"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput�=U�M/�?!�=U�M/�?0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGradu-����?!ښ`�p�?"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInputە�s��?!Q�
�M��?0"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad��z���?!Ƙ�{��?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��2���?!��	":"�?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�?�^aV�?!���M��?0"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFiltera�=?�f�?!���us�?0"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter������?!7:&��4�?0"�
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits�)��� �?!�\Q���?"1
model/Conv1D_3/conv1dConv2DȪ
�1k�?!������?Q      Y@Y��8+?!4@a��15��S@q���s��*@y��u���?"�
both�Your program is MODERATELY input-bound because 6.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�43.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t22.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�13.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 