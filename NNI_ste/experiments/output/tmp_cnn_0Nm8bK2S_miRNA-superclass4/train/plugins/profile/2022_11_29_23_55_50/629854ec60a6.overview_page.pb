�	�VC�"@�VC�"@!�VC�"@      ��!       "{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:�VC�"@���y�?1GtϺF{@I����^�@rEagerKernelExecute 0*	O��n3g@2F
Iterator::Model�E���Դ?!�eu�0�E@)ޓ��ZӬ?1y�JB�U>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�K�A��?!�a�&E@@)xG�j���?1O���i<@:Preprocessing2U
Iterator::Model::ParallelMapV2����B��?!�@�c+@)����B��?1�@�c+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�"R�.�?!I�� X'@)�"R�.�?1I�� X'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateX S�?!޺�X2�0@)�H��rڃ?1=Y�p��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipGq�::��?!��a�L@)8��@��?1e�c���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorB#ظ�]?!W(O�1�@)B#ظ�]?1W(O�1�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa��pɡ?!�%�ӷ2@)Ǆ�K��k?1��|�*�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 5.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�24.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��ic=@Q? ��%�Q@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���y�?���y�?!���y�?      ��!       "	GtϺF{@GtϺF{@!GtϺF{@*      ��!       2      ��!       :	����^�@����^�@!����^�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��ic=@y? ��%�Q@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterU�X�ݼ?!U�X�ݼ?0"1
model/Conv1D_2/conv1dConv2DN)��ж?!�uonQ��?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad���Ħ?!N_��~��?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad����B�?!f����G�?"1
model/Conv1D_3/conv1dConv2D��]z�?!!`�cw�?"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose�M���Π?!�	�_��?"3
model/Conv1D_1/BiasAddBiasAdd�&�鳠?!S�v�o��?"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose_���?!5x��1��?"-
model/Conv1D_1/ReluReluP�����?!��L���?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput��͸�?!�+*+��?0Q      Y@Ym۶m۶)@a�$I�$�U@q����6@ykz�˥�?"�
both�Your program is POTENTIALLY input-bound because 5.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�24.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�23.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 