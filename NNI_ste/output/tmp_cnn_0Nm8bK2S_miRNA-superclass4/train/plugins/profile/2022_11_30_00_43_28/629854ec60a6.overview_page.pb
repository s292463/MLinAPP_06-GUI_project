�	7QKs+<.@7QKs+<.@!7QKs+<.@	Y0�c3� @Y0�c3� @!Y0�c3� @"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC7QKs+<.@���2�?15Lk�h)@I��%P��?Yp��-�?rEagerKernelExecute 0*	V-��sf@2F
Iterator::Model.��M�Ҹ?!B�~�I�J@)�L�ֱ?1�kfC@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat7��-�?!Y��7j49@)$nkϣ?1�q�J�5@:Preprocessing2U
Iterator::Model::ParallelMapV2x���?!J^�	�`.@)x���?1J^�	�`.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate@��>�?!>l�"$�*@)��Z��?1�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�;l"3�?!��&*B2@)�;l"3�?1��&*B2@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�,%�I(�?!�|�6�G@)���<e�?1Z���C@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�X�+��z?!3�P@)�X�+��z?13�P@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap.c}��?!�� �-@)l#�	�h?1�~����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 5.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9Y0�c3� @I`�x�Ϳ+@Q�Eq��U@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���2�?���2�?!���2�?      ��!       "	5Lk�h)@5Lk�h)@!5Lk�h)@*      ��!       2      ��!       :	��%P��?��%P��?!��%P��?B      ��!       J	p��-�?p��-�?!p��-�?R      ��!       Z	p��-�?p��-�?!p��-�?b      ��!       JGPUYY0�c3� @b q`�x�Ϳ+@y�Eq��U@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter
Kr��?!
Kr��?0"1
model/Conv1D_2/conv1dConv2DzC�DT�?!��Z�-�?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputt�M�J;�?!!ހ��]�?0"1
model/Conv1D_3/conv1dConv2D�r�	�?!̺9�-`�?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�v�Ђ��?!��T��?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput�S��̎�?!�.N�
��?0"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilterM��?!I����?0"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��M4.��?!�� �e�?0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad��U��*�?!�{H$|�?"C
%gradient_tape/model/Conv1D_2/ReluGradReluGrad���E�?!}�j�Q6�?Q      Y@Y      )@a     �U@q��Q�1�C@y����̠?"�
both�Your program is POTENTIALLY input-bound because 5.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�8.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�39.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 