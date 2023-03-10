�	3�68(@3�68(@!3�68(@	��di���?��di���?!��di���?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL3�68(@����K��?1��!��#@AS<.�ED�?I7���* @YO ���?rEagerKernelExecute 0*	�MbXe@2F
Iterator::Model��G���?!/��PH@)⬈���?1�`�^$>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat[%X���?!w��X?@)ap��/�?1�6)�,�:@:Preprocessing2U
Iterator::Model::ParallelMapV2ӽN��Ҟ?!��]�C�1@)ӽN��Ҟ?1��]�C�1@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�� x|�?!� �K�@)�� x|�?1� �K�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�.l�V^�?!�\�#��I@)�wak��?1�@�϶@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateQ������?!R�ᕪ*@)���o�?1�u�$�h@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorҧU�f~?!�6�d�@)ҧU�f~?1�6�d�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapJ�O�c�?!��O �9.@)�߽�Ƅh?1c���y�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 1.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�16.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��di���?I�3��1@QƤ�DT@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����K��?����K��?!����K��?      ��!       "	��!��#@��!��#@!��!��#@*      ��!       2	S<.�ED�?S<.�ED�?!S<.�ED�?:	7���* @7���* @!7���* @B      ��!       J	O ���?O ���?!O ���?R      ��!       Z	O ���?O ���?!O ���?b      ��!       JGPUY��di���?b q�3��1@yƤ�DT@�"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad0�����?!0�����?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad�����?!XY[���?"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transposer�q�?!1	G-�}�?"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose�Q���?!�]��${�?"3
model/Conv1D_1/BiasAddBiasAddv]Xr;�?!��� %�?"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose2���+|�?!�%b:���?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterX���w�?!��,���?0"}
^gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer	Transpose�iy��?!R�/Ӄ�?"-
model/Conv1D_1/ReluReluܹh�.+�?!���8)�?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�H:��F�?!�V���?0Q      Y@Ym۶m۶)@a�$I�$�U@q�8�ُ�@y�$4�4ߛ?"�

device�Your program is NOT input-bound because only 1.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�16.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 