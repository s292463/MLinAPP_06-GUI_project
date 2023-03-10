�	}��b�@}��b�@!}��b�@	T��{�@T��{�@!T��{�@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL}��b�@���E��?1m��]�@A�N��Dږ?I1zn��@Y��8h�?rEagerKernelExecute 0*	�"��~>e@2F
Iterator::Model��wF[��?!�~X��G@)�^���?1����=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat#�	�Y�?!a�s�G>@)eQ�E��?1s_x��r:@:Preprocessing2U
Iterator::Model::ParallelMapV2�ο]��?!;�q��1@)�ο]��?1;�q��1@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceC p�ٓ?!����8�&@)C p�ٓ?1����8�&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate:d�w�?!�1Q�`0@)<��~K�?1�݂���@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��c��?!U���`XJ@)�~4�27?1�l���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�%��:�z?!o_��ͤ@)�%��:�z?1o_��ͤ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap:�m½2�?!���=�1@)�?N�0�e?1���:`��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 3.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�40.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9T��{�@IR>��%F@Ql
���I@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���E��?���E��?!���E��?      ��!       "	m��]�@m��]�@!m��]�@*      ��!       2	�N��Dږ?�N��Dږ?!�N��Dږ?:	1zn��@1zn��@!1zn��@B      ��!       J	��8h�?��8h�?!��8h�?R      ��!       Z	��8h�?��8h�?!��8h�?b      ��!       JGPUYT��{�@b qR>��%F@yl
���I@�"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterS� �D�?!S� �D�?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterv��p=�?!�N����?0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad�{73�.�?!d�+�",�?"C
%gradient_tape/model/Conv1D_2/ReluGradReluGrad �1���?!$����?"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGradnQ�����?!�XZ6���?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputI)��?!ê$"�?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradF"��ʠ?!���b��?"1
model/Conv1D_2/conv1dConv2D?ʜ�&q�?!��݆��?"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	TransposeQ�e�ӟ?!��8����?"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose�U�F��?!� U���?Q      Y@Y@n]�G*@a8R4��U@q|���w�@y9ON���?"�

both�Your program is POTENTIALLY input-bound because 3.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�40.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 