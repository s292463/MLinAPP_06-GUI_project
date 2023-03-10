�	��ѫ@��ѫ@!��ѫ@	;���jC@;���jC@!;���jC@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��ѫ@GXT���?1N`:�[�?A^f�(�?I�B��@Y�K�����?rEagerKernelExecute 0*	-����a@2F
Iterator::Modelt��z�Ѯ?!'u���.E@)��c"�?1>RR=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�V
�\�?!��K5|=9@)A�º�?1�Zm��D5@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicef3��J�?!!�8`̣.@)f3��J�?1!�8`̣.@:Preprocessing2U
Iterator::Model::ParallelMapV2|E�^�?! ��n�*@)|E�^�?1 ��n�*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate{.S���?!��ұ~q7@)"S>U��?1��l1? @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���p���?!ڊSK;�L@).;�?l�?1'SJ�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��8Gw?!~�����@)��8Gw?1~�����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�;��~�?! ���'=:@)�k���Dp?18��H]@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 21.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�48.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9;���jC@I��.�Q@Q��|�;@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	GXT���?GXT���?!GXT���?      ��!       "	N`:�[�?N`:�[�?!N`:�[�?*      ��!       2	^f�(�?^f�(�?!^f�(�?:	�B��@�B��@!�B��@B      ��!       J	�K�����?�K�����?!�K�����?R      ��!       Z	�K�����?�K�����?!�K�����?b      ��!       JGPUY;���jC@b q��.�Q@y��|�;@�"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits��.'Aڰ?!��.'Aڰ?"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad}9����?!�d���b�?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter���u�L�?!0U����?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput��B���?!�8\��?0"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput�sˆˢ?!�c�b2�?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�[\ +J�?!4��kvB�?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterz����?!c���X�?0"1
model/Conv1D_1/conv1dConv2D��\j�?�?!�^�,�?"1
model/Conv1D_2/conv1dConv2D�eUj��?!�0���?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad�K�?��?!ڹ)�z�?Q      Y@Y�JG�(@a��7a�U@qz��B@y�U� ���?"�
both�Your program is POTENTIALLY input-bound because 21.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�48.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�36.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 